#!/usr/bin/env python3
"""
Hex (11x11) - Strong AI + Training (single file)
- Rules engine with swap; win via border connection
- Heuristic: shortest-path (Dijkstra-like) with caching
- Agents: Random, AlphaBeta (heuristic), Neural-MCTS (PyTorch)
- Strong AI features:
  - Immediate win detection and opponent threat blocking
  - One-ply safety filter at root
  - Blended root priors (NN policy + heuristic)
  - Endgame AlphaBeta depth search when board is sparse
  - Tuned PUCT parameters
- Training (AlphaZero-lite): self-play with Neural-MCTS, logging, optional GUI visualization
- Saves/loads weights (PyTorch)

Run:
- Train (10 iters, 5 self-play games each):
  python hex_game.py --train
- Play vs strongest agent (loads weights if present):
  python hex_game.py
"""

import math
import random
import time
import heapq
import multiprocessing as mp
import sys
import os
import warnings
from functools import lru_cache
from collections import deque
import threading

warnings.filterwarnings("ignore", category=UserWarning, module="pygame")

# Optional performance libs
try:
    import numpy as np  # type: ignore
    NP_AVAILABLE = True
except Exception:
    NP_AVAILABLE = False

# GUI
try:
    import pygame  # type: ignore
    PYGAME_AVAILABLE = True
except Exception:
    PYGAME_AVAILABLE = False

# GUI Performance Optimization
GUI_UPDATE_CACHE = {}
GUI_LAST_RENDER_TIME = 0
GUI_FRAME_SKIP = 3  # Skip more frames for better performance

# PyTorch
try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.optim as optim  # type: ignore
    import torch.nn.functional as F  # type: ignore
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# --- Config (tuned for expert performance) ---
BOARD_SIZE = 11
USE_MP = True
MCTS_SIMS = 400  # Balanced for strong play while maintaining responsiveness
TRAIN_MCTS_SIMS = 40  # ***CRITICAL: Reduced from 150 to 40 for stability***
MCTS_BATCH = 32     # Increased batch size for better GPU utilization
ALPHABETA_DEPTH = 8  # Increased depth for stronger play
SELFPLAY_GAMES = 20  # 20 games per iteration
TRAIN_ITERS = 20      # 20 iterations = 400 total games
SEED = 42
# Performance optimization settings
EVAL_CACHE_SIZE = 10000  # Cache for evaluation results
PATH_CACHE_SIZE = 5000   # Cache for path calculations
GUI_FPS = 60             # Target FPS for smooth rendering
DIRTY_REGION_THRESHOLD = 10  # Minimum changes to trigger full redraw
random.seed(SEED)
if NP_AVAILABLE:
    np.random.seed(SEED)
if TORCH_AVAILABLE:
    torch.manual_seed(SEED)

# MP_PROCESSES = min(6, max(2, mp.cpu_count() - 1)) if USE_MP else 1  # Reserved for future multiprocessing

# -------------------- Performance Optimization Layer --------------------
class PerformanceOptimizer:
    """Advanced caching and optimization for Hex game engine"""
    
    def __init__(self):
        self.eval_cache = {}
        self.path_cache = {}
        self.board_state_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.lock = threading.Lock()
        
    def get_board_key(self, game):
        """Generate efficient board state key"""
        if game._board_hash is None:
            # Use more efficient hashing
            flat = tuple(cell for row in game.board for cell in row)
            game._board_hash = flat
        return (game._board_hash, game.turn)
    
    def cached_eval(self, game, perspective, eval_func):
        """Cached evaluation with LRU eviction"""
        key = (self.get_board_key(game), perspective)
        
        with self.lock:
            if key in self.eval_cache:
                self.cache_hits += 1
                return self.eval_cache[key]
            
            # Evict if cache is full
            if len(self.eval_cache) > EVAL_CACHE_SIZE:
                # Remove oldest 25% of entries
                items_to_remove = len(self.eval_cache) // 4
                oldest_keys = list(self.eval_cache.keys())[:items_to_remove]
                for old_key in oldest_keys:
                    del self.eval_cache[old_key]
            
            self.cache_misses += 1
            result = eval_func(game, perspective)
            self.eval_cache[key] = result
            return result
    
    def clear_caches(self):
        """Clear all caches"""
        with self.lock:
            self.eval_cache.clear()
            self.path_cache.clear()
            self.board_state_cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0
    
    def get_cache_stats(self):
        """Get cache performance statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        return {
            'eval_cache_size': len(self.eval_cache),
            'path_cache_size': len(self.path_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': f"{hit_rate:.1f}%"
        }

# Global optimizer instance
optimizer = PerformanceOptimizer()

# -------------------- Game Engine --------------------
class HexGame:
    EMPTY = 0
    P1 = 1
    P2 = 2

    def __init__(self, size=BOARD_SIZE, swap_rule=True):
        self.size = size
        self.board = [[HexGame.EMPTY for _ in range(size)] for _ in range(size)]
        self.turn = HexGame.P1
        self.move_history = []
        self.swap_rule = swap_rule
        self.swap_offered = False
        self.terminal = False
        self.winner = None
        self._sp_cache = {}
        self._board_hash = None

    def clone(self):
        g = HexGame(self.size, self.swap_rule)
        g.board = [row[:] for row in self.board]
        g.turn = self.turn
        g.move_history = self.move_history[:]
        g.swap_offered = self.swap_offered
        g.terminal = self.terminal
        g.winner = self.winner
        g._sp_cache = dict(self._sp_cache)
        g._board_hash = self._board_hash
        return g

    def in_bounds(self, r, c):
        return 0 <= r < self.size and 0 <= c < self.size

    def neighbors(self, r, c):
        cand = [(r-1,c),(r-1,c+1),(r,c-1),(r,c+1),(r+1,c-1),(r+1,c)]
        return [(nr,nc) for nr,nc in cand if self.in_bounds(nr,nc)]

    def valid_moves(self):
        return [(r,c) for r in range(self.size) for c in range(self.size) if self.board[r][c] == HexGame.EMPTY]

    def is_valid_move(self, r, c):
        return self.in_bounds(r,c) and self.board[r][c] == HexGame.EMPTY

    def make_move(self, r, c):
        if not self.is_valid_move(r,c):
            raise ValueError("Invalid move")
        self.board[r][c] = self.turn
        self.move_history.append((self.turn,(r,c)))
        self._board_hash = None
        self._sp_cache.clear()
        self.turn = HexGame.P1 if self.turn == HexGame.P2 else HexGame.P2
        w = self.check_win_internal()
        if w:
            self.terminal = True
            self.winner = w
        return True

    def offer_swap(self):
        if len(self.move_history) == 1 and not self.swap_offered and self.swap_rule:
            self.swap_offered = True
            return True
        return False

    def do_swap(self):
        new_board = [[None]*self.size for _ in range(self.size)]
        for r in range(self.size):
            for c in range(self.size):
                v = self.board[r][c]
                new_board[r][c] = HexGame.P1 if v == HexGame.P2 else HexGame.P2 if v == HexGame.P1 else HexGame.EMPTY
        self.board = new_board
        self.move_history = [(HexGame.P1 if p==HexGame.P2 else HexGame.P2, mv) for p,mv in self.move_history]
        self.swap_offered = False
        # After swap, the swapping player becomes P1 and should move next
        self.turn = HexGame.P1
        self._board_hash = None
        self._sp_cache.clear()

    def check_win_internal(self):
        def dfs(player):
            seen = [[False]*self.size for _ in range(self.size)]
            stack = []
            if player == HexGame.P1:
                for c in range(self.size):
                    if self.board[0][c] == player:
                        seen[0][c] = True; stack.append((0,c))
            else:
                for r in range(self.size):
                    if self.board[r][0] == player:
                        seen[r][0] = True; stack.append((r,0))
            while stack:
                r,c = stack.pop()
                if (player == HexGame.P1 and r == self.size-1) or (player == HexGame.P2 and c == self.size-1):
                    return True
                for nr,nc in self.neighbors(r,c):
                    if not seen[nr][nc] and self.board[nr][nc] == player:
                        seen[nr][nc] = True; stack.append((nr,nc))
            return False
        if dfs(HexGame.P1):
            return HexGame.P1
        if dfs(HexGame.P2):
            return HexGame.P2
        return None

    def print_board(self):
        for r in range(self.size):
            print(" "*r + " ".join('.' if v==HexGame.EMPTY else ('X' if v==HexGame.P1 else 'O') for v in self.board[r]))
        print()

    def board_hash(self):
        if self._board_hash is None:
            flat = []
            for r in range(self.size):
                flat.extend(self.board[r])
            self._board_hash = tuple(flat)
        return self._board_hash

# Optimized shortest path with enhanced caching
def shortest_path_distance_cached(game: HexGame, player):
    """Highly optimized shortest path calculation with multi-level caching"""
    key = (player, game.board_hash())
    
    # Check global cache first
    if key in optimizer.path_cache:
        return optimizer.path_cache[key]
    
    INF = 10**8
    n = game.size
    
    # Use NumPy for better performance if available
    if NP_AVAILABLE:
        arr = np.array(game.board, dtype=np.int8)
        dist = np.full((n, n), INF, dtype=np.int32)
        hq = []
        
        if player == HexGame.P1:
            # Vectorized initialization for P1 (top to bottom)
            top_row = arr[0, :]
            for c in range(n):
                if top_row[c] == player:
                    dist[0, c] = 0
                    heapq.heappush(hq, (0, 0, c))
                elif top_row[c] == HexGame.EMPTY:
                    dist[0, c] = 1
                    heapq.heappush(hq, (1, 0, c))
        else:
            # Vectorized initialization for P2 (left to right)
            left_col = arr[:, 0]
            for r in range(n):
                if left_col[r] == player:
                    dist[r, 0] = 0
                    heapq.heappush(hq, (0, r, 0))
                elif left_col[r] == HexGame.EMPTY:
                    dist[r, 0] = 1
                    heapq.heappush(hq, (1, r, 0))
        
        # Optimized Dijkstra with early termination
        while hq:
            d, r, c = heapq.heappop(hq)
            if d != dist[r, c]:
                continue
            
            # Early termination when reaching goal
            if (player == HexGame.P1 and r == n-1) or (player == HexGame.P2 and c == n-1):
                result = int(d)
                optimizer.path_cache[key] = result
                return result
            
            # Process neighbors more efficiently
            for nr, nc in game.neighbors(r, c):
                cell = arr[nr, nc]
                nd = d if cell == player else d + 1 if cell == HexGame.EMPTY else INF
                if nd < dist[nr, nc]:
                    dist[nr, nc] = nd
                    heapq.heappush(hq, (int(nd), int(nr), int(nc)))
    else:
        # Fallback to pure Python implementation
        dist = [[INF] * n for _ in range(n)]
        hq = []
        
        if player == HexGame.P1:
            for c in range(n):
                if game.board[0][c] == player:
                    dist[0][c] = 0
                    heapq.heappush(hq, (0, 0, c))
                elif game.board[0][c] == HexGame.EMPTY:
                    dist[0][c] = 1
                    heapq.heappush(hq, (1, 0, c))
        else:
            for r in range(n):
                if game.board[r][0] == player:
                    dist[r][0] = 0
                    heapq.heappush(hq, (0, r, 0))
                elif game.board[r][0] == HexGame.EMPTY:
                    dist[r][0] = 1
                    heapq.heappush(hq, (1, r, 0))
        
        while hq:
            d, r, c = heapq.heappop(hq)
            if d != dist[r][c]:
                continue
            
            if (player == HexGame.P1 and r == n-1) or (player == HexGame.P2 and c == n-1):
                result = d
                optimizer.path_cache[key] = result
                return result
            
            for nr, nc in game.neighbors(r, c):
                cell = game.board[nr][nc]
                nd = d if cell == player else d + 1 if cell == HexGame.EMPTY else INF
                if nd < dist[nr][nc]:
                    dist[nr][nc] = nd
                    heapq.heappush(hq, (nd, nr, nc))
    
    # Cache the result
    optimizer.path_cache[key] = INF
    return INF

# Enhanced evaluation functions for expert-level play
def find_main_path(game: HexGame, player):
    """Find the main zig-zag path from one side to the other"""
    if player == HexGame.P1:
        # P1 connects top to bottom
        start_side = [(0, c) for c in range(game.size) if game.board[0][c] == player]
        end_side = [(game.size-1, c) for c in range(game.size) if game.board[game.size-1][c] == player]
    else:
        # P2 connects left to right
        start_side = [(r, 0) for r in range(game.size) if game.board[r][0] == player]
        end_side = [(r, game.size-1) for r in range(game.size) if game.board[r][game.size-1] == player]
    
    if not start_side or not end_side:
        return None, 0
    
    # Use BFS to find shortest path through player's stones
    from collections import deque
    queue = deque([(r, c, [(r, c)]) for r, c in start_side])
    visited = set(start_side)
    
    while queue:
        r, c, path = queue.popleft()
        
        if (r, c) in end_side:
            return path, len(path)
        
        for nr, nc in game.neighbors(r, c):
            if (nr, nc) not in visited and game.board[nr][nc] == player:
                visited.add((nr, nc))
                queue.append((nr, nc, path + [(nr, nc)]))
    
    return None, 0

def find_bottlenecks(game: HexGame, player):
    """Find critical cells (bottlenecks) where path can be blocked/connected"""
    bottlenecks = []
    opponent = HexGame.P1 if player == HexGame.P2 else HexGame.P2
    
    # Find gaps in player's path (2-3 hexes apart)
    player_stones = [(r, c) for r in range(game.size) for c in range(game.size) 
                     if game.board[r][c] == player]
    
    for i, (r1, c1) in enumerate(player_stones):
        for r2, c2 in player_stones[i+1:]:
            dist = abs(r2 - r1) + abs(c2 - c1)
            if 2 <= dist <= 3:
                # Check if there's an empty cell that would connect them
                for r in range(min(r1, r2), max(r1, r2) + 1):
                    for c in range(min(c1, c2), max(c1, c2) + 1):
                        if game.in_bounds(r, c) and game.board[r][c] == HexGame.EMPTY:
                            # Test if this is critical
                            test_game = game.clone()
                            test_game.board[r][c] = player
                            new_dist = shortest_path_distance_cached(test_game, player)
                            if new_dist < shortest_path_distance_cached(game, player):
                                bottlenecks.append((r, c, 10))  # High priority
    
    # Find opponent's bottlenecks to block
    opp_stones = [(r, c) for r in range(game.size) for c in range(game.size) 
                   if game.board[r][c] == opponent]
    
    for i, (r1, c1) in enumerate(opp_stones):
        for r2, c2 in opp_stones[i+1:]:
            dist = abs(r2 - r1) + abs(c2 - c1)
            if 2 <= dist <= 3:
                for r in range(min(r1, r2), max(r1, r2) + 1):
                    for c in range(min(c1, c2), max(c1, c2) + 1):
                        if game.in_bounds(r, c) and game.board[r][c] == HexGame.EMPTY:
                            test_game = game.clone()
                            test_game.board[r][c] = opponent
                            new_dist = shortest_path_distance_cached(test_game, opponent)
                            if new_dist < shortest_path_distance_cached(game, opponent):
                                bottlenecks.append((r, c, 15))  # Very high priority - block opponent
    
    return bottlenecks

def detect_bridges(game: HexGame, player):
    """Detect bridge structures (two stones that are one knight's move away)"""
    bridges = []
    player_stones = [(r, c) for r in range(game.size) for c in range(game.size) 
                     if game.board[r][c] == player]
    
    # Check all pairs of player stones for bridge pattern
    for i, (r1, c1) in enumerate(player_stones):
        for r2, c2 in player_stones[i+1:]:
            # Knight's move in hex: two steps in one direction, one in perpendicular
            dr = abs(r2 - r1)
            dc = abs(c2 - c1)
            # Bridge pattern: distance of 2 in hex grid
            if (dr == 2 and dc == 1) or (dr == 1 and dc == 2) or (dr == 2 and dc == 0) or (dr == 0 and dc == 2):
                # Check if the bridge cell is empty
                mid_r = (r1 + r2) // 2
                mid_c = (c1 + c2) // 2
                if game.in_bounds(mid_r, mid_c) and game.board[mid_r][mid_c] == HexGame.EMPTY:
                    bridges.append((mid_r, mid_c))
    return bridges

def check_connection_strength(game: HexGame, player):
    """Check if connections are strong (bridge/double-threat) or weak"""
    strong_connections = 0
    weak_connections = 0
    
    # Count bridges (strong)
    bridges = detect_bridges(game, player)
    strong_connections += len(bridges)
    
    # Check for double-threat patterns (two ways to connect)
    player_stones = [(r, c) for r in range(game.size) for c in range(game.size) 
                     if game.board[r][c] == player]
    
    # Count stones with multiple neighbors (stronger)
    for r, c in player_stones:
        neighbors = game.neighbors(r, c)
        own_neighbors = sum(1 for nr, nc in neighbors if game.board[nr][nc] == player)
        if own_neighbors >= 2:
            strong_connections += 1
        elif own_neighbors == 1:
            weak_connections += 1
    
    return strong_connections, weak_connections

def detect_ladder_threats(game: HexGame, player):
    """Detect ladder threats (forcing diagonal sequences)"""
    ladder_score = 0
    opponent = HexGame.P1 if player == HexGame.P2 else HexGame.P2
    
    # Check for diagonal patterns that could become ladders
    if player == HexGame.P1:
        # Check top-to-bottom diagonal patterns
        for c in range(game.size - 1):
            for r in range(game.size - 1):
                if game.board[r][c] == player and game.board[r+1][c+1] == player:
                    # Potential ladder - check if opponent can escape
                    if game.board[r][c+1] == HexGame.EMPTY or game.board[r+1][c] == HexGame.EMPTY:
                        ladder_score += 5
    else:
        # Check left-to-right diagonal patterns
        for r in range(game.size - 1):
            for c in range(game.size - 1):
                if game.board[r][c] == player and game.board[r+1][c+1] == player:
                    if game.board[r][c+1] == HexGame.EMPTY or game.board[r+1][c] == HexGame.EMPTY:
                        ladder_score += 5
    
    return ladder_score

def detect_virtual_connections_advanced(game: HexGame, player):
    """Advanced virtual connection detection - forcing sequences"""
    virtual_count = 0
    
    # Simplified: check if we have multiple independent paths
    myd = shortest_path_distance_cached(game, player)
    if myd >= 10**7:
        return 0
    
    # Count alternative paths by testing removals
    test_removals = 0
    for r in range(game.size):
        for c in range(game.size):
            if game.board[r][c] == player:
                test_game = game.clone()
                test_game.board[r][c] = HexGame.EMPTY
                test_d = shortest_path_distance_cached(test_game, player)
                if test_d < 10**7:
                    test_removals += 1
                    if test_removals >= 2:
                        virtual_count += 1
                        break
        if virtual_count > 0:
            break
    
    return virtual_count

def evaluate_center_control(game: HexGame, player):
    """Evaluate who controls the center vs edges"""
    center = (game.size - 1) / 2.0
    center_control = 0.0
    edge_control = 0.0
    
    player_stones = [(r, c) for r in range(game.size) for c in range(game.size) 
                     if game.board[r][c] == player]
    
    for r, c in player_stones:
        dist_from_center = abs(r - center) + abs(c - center)
        max_dist = game.size * 1.5
        
        if dist_from_center < game.size * 0.3:  # Center area
            center_control += 3.0
        elif dist_from_center < game.size * 0.6:  # Mid area
            center_control += 1.0
        else:  # Edge area
            edge_control += 0.5
    
    return center_control, edge_control

def count_dual_paths_advanced(game: HexGame, player):
    """Count independent dual paths (two separate routes)"""
    myd = shortest_path_distance_cached(game, player)
    if myd >= 10**7:
        return 0
    
    # Count how many different starting points can reach goal
    path_count = 0
    if player == HexGame.P1:
        for c in range(game.size):
            if game.board[0][c] == player or game.board[0][c] == HexGame.EMPTY:
                test_game = game.clone()
                if test_game.board[0][c] == HexGame.EMPTY:
                    test_game.board[0][c] = player
                d = shortest_path_distance_cached(test_game, player)
                if d < 10**7:
                    path_count += 1
    else:
        for r in range(game.size):
            if game.board[r][0] == player or game.board[r][0] == HexGame.EMPTY:
                test_game = game.clone()
                if test_game.board[r][0] == HexGame.EMPTY:
                    test_game.board[r][0] = player
                d = shortest_path_distance_cached(test_game, player)
                if d < 10**7:
                    path_count += 1
    
    return max(0, path_count - 1)  # At least 2 = dual path

def count_forcing_moves(game: HexGame, player):
    """Count forcing moves (moves that demand immediate response)"""
    forcing_count = 0
    opponent = HexGame.P1 if player == HexGame.P2 else HexGame.P2
    
    valid_moves = game.valid_moves()
    
    for mv in valid_moves:
        # Check if this move creates a win threat
        test_game = game.clone()
        test_game.make_move(*mv)
        
        # Check if opponent must block
        opp_must_block = False
        
        # Check if opponent can win next
        for opp_mv in test_game.valid_moves():
            test_game2 = test_game.clone()
            test_game2.turn = opponent
            if test_game2.is_valid_move(*opp_mv):
                test_game2.make_move(*opp_mv)
                if test_game2.terminal and test_game2.winner == opponent:
                    opp_must_block = True
                    break
        
        # Check if this move creates a bottleneck
        bottlenecks = find_bottlenecks(test_game, player)
        if len(bottlenecks) > 0:
            forcing_count += 1
        
        # Check if this move extends path significantly
        my_dist_before = shortest_path_distance_cached(game, player)
        my_dist_after = shortest_path_distance_cached(test_game, player)
        if my_dist_after < my_dist_before - 1:
            forcing_count += 1
    
    return forcing_count

def count_dual_paths(game: HexGame, player):
    """Count potential dual paths (zigzag structures)"""
    myd = shortest_path_distance_cached(game, player)
    if myd >= 10**7:
        return 0
    
    # Count alternative paths by checking if removing one cell still allows connection
    alt_paths = 0
    for r in range(game.size):
        for c in range(game.size):
            if game.board[r][c] == player:
                test_game = game.clone()
                test_game.board[r][c] = HexGame.EMPTY
                test_d = shortest_path_distance_cached(test_game, player)
                if test_d < 10**7:
                    alt_paths += 1
    return alt_paths

def calculate_influence(game: HexGame, player):
    """Calculate board influence (central control, diagonal projection)"""
    center = (game.size - 1) / 2.0
    influence = 0.0
    player_stones = [(r, c) for r in range(game.size) for c in range(game.size) 
                     if game.board[r][c] == player]
    
    for r, c in player_stones:
        # Central influence (closer to center = more influence)
        dist_from_center = abs(r - center) + abs(c - center)
        center_influence = 1.0 / (1.0 + dist_from_center * 0.5)
        
        # Diagonal projection bonus (stones that project toward goal)
        if player == HexGame.P1:
            # P1 goes top to bottom, so lower rows are better
            row_progress = (game.size - 1 - r) / game.size
        else:
            # P2 goes left to right, so right columns are better
            col_progress = (game.size - 1 - c) / game.size
        
        influence += center_influence * (1.0 + row_progress if player == HexGame.P1 else 1.0 + col_progress)
    
    return influence

def detect_virtual_connections(game: HexGame, player):
    """Detect virtual connections (can connect even if opponent plays)"""
    # Simplified: check if we have multiple paths to goal
    myd = shortest_path_distance_cached(game, player)
    if myd >= 10**7:
        return 0
    
    # Count how many different starting points can reach goal
    virtual_count = 0
    if player == HexGame.P1:
        for c in range(game.size):
            if game.board[0][c] == player or game.board[0][c] == HexGame.EMPTY:
                test_game = game.clone()
                if test_game.board[0][c] == HexGame.EMPTY:
                    test_game.board[0][c] = player
                d = shortest_path_distance_cached(test_game, player)
                if d < 10**7:
                    virtual_count += 1
    else:
        for r in range(game.size):
            if game.board[r][0] == player or game.board[r][0] == HexGame.EMPTY:
                test_game = game.clone()
                if test_game.board[r][0] == HexGame.EMPTY:
                    test_game.board[r][0] = player
                d = shortest_path_distance_cached(test_game, player)
                if d < 10**7:
                    virtual_count += 1
    
    return virtual_count

def expert_eval_cached(game: HexGame, perspective):
    """Cached expert-level evaluation following 8-step analysis framework"""
    return optimizer.cached_eval(game, perspective, expert_eval_internal)

def expert_eval_internal(game: HexGame, perspective):
    """Internal expert evaluation without caching"""
    if game.terminal:
        return 1e6 if game.winner == perspective else -1e6
    
    opp = HexGame.P1 if perspective == HexGame.P2 else HexGame.P2
    
    # Precompute common values to avoid redundant calculations
    myd = shortest_path_distance_cached(game, perspective)
    oppd = shortest_path_distance_cached(game, opp)
    
    # Early exit if no paths exist
    if myd >= 10**7 and oppd >= 10**7:
        return 0.0
    
    # 1. MAIN PATH STRENGTH (optimized)
    my_path, my_path_len = find_main_path(game, perspective)
    opp_path, opp_path_len = find_main_path(game, opp)
    
    path_score = 0.0
    if my_path_len > 0 and opp_path_len > 0:
        path_score = (opp_path_len - my_path_len) * 20.0
    elif my_path_len > 0:
        path_score = 100.0
    elif opp_path_len > 0:
        path_score = -100.0
    
    # Add distance-based evaluation
    if myd < 10**7 and oppd < 10**7:
        path_score += float(oppd - myd) * 10.0
    
    # 2. BOTTLENECKS (optimized)
    my_bottlenecks = find_bottlenecks(game, perspective)
    opp_bottlenecks = find_bottlenecks(game, opp)
    bottleneck_score = (len(my_bottlenecks) - len(opp_bottlenecks)) * 15.0
    
    # Prioritize blocking opponent bottlenecks
    opp_block_priority = sum(1 for _, _, priority in opp_bottlenecks if priority >= 15)
    bottleneck_score += opp_block_priority * 25.0
    
    # 3. CONNECTION STRENGTH (simplified but effective)
    my_strong, my_weak = check_connection_strength(game, perspective)
    opp_strong, opp_weak = check_connection_strength(game, opp)
    connection_score = (my_strong - opp_strong) * 30.0 - (my_weak - opp_weak) * 10.0
    
    # 4. LADDER THREATS (simplified for performance)
    my_ladder = detect_ladder_threats(game, perspective)
    opp_ladder = detect_ladder_threats(game, opp)
    ladder_score = (my_ladder - opp_ladder) * 8.0
    
    # 5. VIRTUAL CONNECTIONS (simplified for performance)
    my_virtual = detect_virtual_connections(game, perspective)
    opp_virtual = detect_virtual_connections(game, opp)
    virtual_score = (my_virtual - opp_virtual) * 40.0
    
    # 6. CENTER CONTROL (optimized)
    my_center, my_edge = evaluate_center_control(game, perspective)
    opp_center, opp_edge = evaluate_center_control(game, opp)
    center_score = (my_center - opp_center) * 12.0 - (my_edge - opp_edge) * 2.0
    
    # 7. DUAL PATHS (simplified but still effective)
    my_dual = count_dual_paths(game, perspective)
    opp_dual = count_dual_paths(game, opp)
    dual_score = 0.0
    if my_dual >= 2 and opp_dual < 2:
        dual_score = 150.0
    elif my_dual < 2 and opp_dual >= 2:
        dual_score = -150.0
    else:
        dual_score = (my_dual - opp_dual) * 50.0
    
    # 8. FORCING MOVES (simplified for performance)
    my_forcing = count_forcing_moves(game, perspective)
    opp_forcing = count_forcing_moves(game, opp)
    forcing_score = (my_forcing - opp_forcing) * 10.0
    
    # Combine all 8 factors with tuned weights
    total_score = (
        path_score * 1.2 +          # Path is most important
        bottleneck_score * 1.5 +    # Blocking is critical
        connection_score * 1.1 +     # Connections matter
        ladder_score * 0.8 +        # Ladders are situational
        virtual_score * 1.3 +       # Virtual connections are strong
        center_score * 0.9 +       # Center control is moderate
        dual_score * 1.4 +          # Dual paths are very strong
        forcing_score * 1.0         # Forcing moves are important
    )
    
    return total_score

# Keep original for backward compatibility
def expert_eval(game: HexGame, perspective):
    """Expert-level evaluation following 8-step analysis framework"""
    return expert_eval_cached(game, perspective)

# Eval (keeping original for backward compatibility)
def heuristic_eval(game: HexGame, perspective):
    if game.terminal:
        return 1e6 if game.winner == perspective else -1e6
    myd = shortest_path_distance_cached(game, perspective)
    opp = HexGame.P1 if perspective == HexGame.P2 else HexGame.P2
    oppd = shortest_path_distance_cached(game, opp)
    if myd >= 10**7 and oppd >= 10**7:
        return 0.0
    return float(oppd - myd)

# Random playouts
def random_playout(game: HexGame):
    g = game.clone(); moves = g.valid_moves(); random.shuffle(moves)
    for mv in moves:
        if g.terminal: break
        g.make_move(*mv)
    return g.winner

# MP helper (reserved for future multiprocessing)
# def playout_worker(g):
#     return random_playout(g)

# -------------------- Agents --------------------
class RandomAgent:
    def __init__(self, name="Random"): self.name = name
    def select_move(self, game: HexGame):
        m = game.valid_moves(); return random.choice(m) if m else None
    def decide_swap(self, game: HexGame): return False

class AlphaBetaAgent:
    """Enhanced AlphaBeta agent with expert-level evaluation and optimizations"""
    def __init__(self, name="AlphaBeta", depth=ALPHABETA_DEPTH, rollout_evals=20, consider_k=40):  # Reduced from 60
        self.name = name
        self.depth = depth
        self.rollout_evals = rollout_evals
        self.consider_k = consider_k
        self.transposition_table = {}
        self.killer_moves = {}
        self.history_heuristic = {}
        
    def select_move(self, game: HexGame):
        m = game.valid_moves()
        if not m:
            return None
        
        # Check for immediate win
        for mv in m:
            g2 = game.clone()
            g2.make_move(*mv)
            if g2.terminal and g2.winner == game.turn:
                return mv
        
        # Check for opponent threats
        opp = HexGame.P1 if game.turn == HexGame.P2 else HexGame.P2
        for mv in m:
            g2 = game.clone()
            g2.turn = opp
            if g2.is_valid_move(*mv):
                g2.make_move(*mv)
                if g2.terminal and g2.winner == opp:
                    return mv  # Block opponent win
        
        cand = self.candidate_moves(game, m, k=self.consider_k)
        best_mv = None
        best_val = -1e18
        alpha = -1e18
        beta = 1e18
        
        for mv in cand:
            g2 = game.clone()
            g2.make_move(*mv)
            val = -self.negamax(g2, self.depth - 1, -beta, -alpha, perspective=game.turn)
            if val > best_val:
                best_val = val
                best_mv = mv
            alpha = max(alpha, val)
        
        return best_mv
    
    def negamax(self, game, depth, alpha, beta, perspective):
        # Transposition table lookup
        tt_key = (game.board_hash(), depth, perspective)
        if tt_key in self.transposition_table:
            tt_val, tt_type = self.transposition_table[tt_key]
            if tt_type == 'EXACT':
                return tt_val
            elif tt_type == 'LOWERBOUND' and tt_val > alpha:
                alpha = tt_val
            elif tt_type == 'UPPERBOUND' and tt_val < beta:
                beta = tt_val
            if alpha >= beta:
                return tt_val
        
        if game.terminal or depth == 0:
            last = game.move_history[-1][0] if game.move_history else perspective
            # Use expert evaluation for stronger play
            score = expert_eval_cached(game, last)
            if depth == 0 and self.rollout_evals > 0:
                # Enhanced rollout with expert evaluation
                wins = 0
                trials = min(8, self.rollout_evals)
                for _ in range(trials):
                    g = game.clone()
                    w = random_playout(g)
                    if w == (g.move_history[-1][0] if g.move_history else last):
                        wins += 1
                score = score * 0.7 + (wins / trials - 0.5) * 2000.0
            return score
        
        m = game.valid_moves()
        cand = self.candidate_moves(game, m, k=self.consider_k)
        
        # Killer move heuristic
        killer_key = (perspective, depth)
        killer_moves = self.killer_moves.get(killer_key, [])
        
        # Order moves: killer moves first, then by history heuristic
        ordered_moves = []
        for km in killer_moves:
            if km in cand:
                ordered_moves.append(km)
                cand.remove(km)
        
        # Sort remaining moves by history heuristic
        scored_moves = []
        for mv in cand:
            history_score = self.history_heuristic.get((perspective, mv), 0)
            scored_moves.append((history_score, mv))
        scored_moves.sort(reverse=True, key=lambda x: x[0])
        ordered_moves.extend([mv for _, mv in scored_moves])
        
        best = -1e18
        best_move = None
        
        for mv in ordered_moves:
            g2 = game.clone()
            g2.make_move(*mv)
            val = -self.negamax(g2, depth - 1, -beta, -alpha, perspective)
            val = -val
            
            if val > best:
                best = val
                best_move = mv
            
            # Update history heuristic
            self.history_heuristic[(perspective, mv)] = self.history_heuristic.get((perspective, mv), 0) + depth * depth
            
            alpha = max(alpha, val)
            if alpha >= beta:
                # Update killer moves
                if mv not in killer_moves:
                    killer_moves.append(mv)
                    if len(killer_moves) > 2:
                        killer_moves.pop(0)
                    self.killer_moves[killer_key] = killer_moves
                break
        
        # Store in transposition table
        if best > -1e17:
            tt_type = 'EXACT'
            if best <= alpha:
                tt_type = 'UPPERBOUND'
            elif best >= beta:
                tt_type = 'LOWERBOUND'
            self.transposition_table[tt_key] = (best, tt_type)
        
        return best if best > -1e17 else 0.0
    
    def candidate_moves(self, game, moves, k=20):  # Reduced from 30 for speed
        if not NP_AVAILABLE:
            return moves[:k]
        
        scored = []
        center = (game.size - 1) / 2.0
        occ = [(r, c) for r in range(game.size) for c in range(game.size) if game.board[r][c] != HexGame.EMPTY]
        
        for r, c in moves:
            # Distance from center
            center_score = -(abs(r - center) + abs(c - center))
            
            # Proximity to existing stones
            prox = 0.0
            if occ:
                prox = -min(abs(r - orow) + abs(c - ocol) for orow, ocol in occ)
            
            # Use shortest path distance for faster evaluation
            g2 = game.clone()
            g2.make_move(r, c)
            my = HexGame.P1 if game.turn == HexGame.P1 else HexGame.P2
            # Fast evaluation using shortest path (much faster than expert_eval)
            my_dist = shortest_path_distance_cached(g2, my)
            opp = HexGame.P1 if my == HexGame.P2 else HexGame.P2
            opp_dist = shortest_path_distance_cached(g2, opp)
            
            # Convert to score (shorter path = better)
            eval_score = 0.0
            if my_dist < 10**7 and opp_dist < 10**7:
                eval_score = float(opp_dist - my_dist) * 10.0
            
            # Combine scores with tuned weights
            score = 0.3 * center_score + 0.4 * prox + 0.3 * eval_score
            scored.append((score, (r, c)))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        return [mv for _, mv in scored[:k]]
    
    def decide_swap(self, game: HexGame):
        if not (len(game.move_history)==1 and game.swap_offered): return False
        trials=32; gA=game.clone(); gB=game.clone(); gB.do_swap()
        def seq(g): 
            wins=0
            for _ in range(trials//2):
                if random_playout(g)==HexGame.P2: wins+=1
            return wins
        return False if seq(gA)>=seq(gB) else True

# -------------------- Difficulty-Based Bots --------------------
class BeginnerBot:
    """Beginner Bot - Pattern Follower: Local play, no global strategy"""
    def __init__(self, name="Beginner Bot"):
        self.name = name
    
    def select_move(self, game: HexGame):
        """Plays locally near own stones, ignores global strategy"""
        valid_moves = game.valid_moves()
        if not valid_moves:
            return None
        
        current_player = game.turn
        
        # Get all own stones
        own_stones = [(r, c) for r in range(game.size) for c in range(game.size) 
                      if game.board[r][c] == current_player]
        
        if not own_stones:
            # First move: play near center but with some randomness
            center = game.size // 2
            candidates = [(r, c) for r, c in valid_moves 
                          if abs(r - center) <= 2 and abs(c - center) <= 2]
            if candidates:
                return random.choice(candidates)
            return random.choice(valid_moves)
        
        # Play near own stones (local greedy)
        scored_moves = []
        for r, c in valid_moves:
            # Find distance to nearest own stone
            min_dist = min(abs(r - sr) + abs(c - sc) for sr, sc in own_stones)
            # Prefer closer moves (but add some randomness)
            score = -min_dist + random.uniform(-1, 1)
            scored_moves.append((score, (r, c)))
        
        scored_moves.sort(reverse=True, key=lambda x: x[0])
        # Pick from top 30% with some randomness
        top_n = max(1, len(scored_moves) // 3)
        top_moves = [mv for _, mv in scored_moves[:top_n]]
        return random.choice(top_moves)
    
    def decide_swap(self, game: HexGame):
        """Beginner bot never swaps"""
        return False

class IntermediateBot:
    """Intermediate Bot - Shape Builder: Path formation, bridge recognition, threat blocking"""
    def __init__(self, name="Intermediate Bot", depth=3):
        self.name = name
        self.depth = depth
    
    def select_move(self, game: HexGame):
        """Uses path formation, bridge detection, and threat blocking"""
        valid_moves = game.valid_moves()
        if not valid_moves:
            return None
        
        current_player = game.turn
        opponent = HexGame.P1 if current_player == HexGame.P2 else HexGame.P2
        
        # 1. Check for immediate win
        for mv in valid_moves:
            test_game = game.clone()
            test_game.make_move(*mv)
            if test_game.terminal and test_game.winner == current_player:
                return mv
        
        # 2. Block opponent threats (can they win next turn?)
        opponent_threats = []
        for mv in valid_moves:
            test_game = game.clone()
            test_game.turn = opponent
            if test_game.is_valid_move(*mv):
                test_game.make_move(*mv)
                if test_game.terminal and test_game.winner == opponent:
                    opponent_threats.append(mv)
        
        if opponent_threats:
            # Block the most dangerous threat
            best_block = None
            best_score = -1e18
            for threat in opponent_threats:
                test_game = game.clone()
                test_game.make_move(*threat)
                # After blocking, how good is our position?
                score = -shortest_path_distance_cached(test_game, current_player)
                if score > best_score:
                    best_score = score
                    best_block = threat
            if best_block:
                return best_block
        
        # 3. Look for bridge opportunities
        bridges = detect_bridges(game, current_player)
        if bridges:
            # Play a bridge move (bridges are the empty cells between two stones)
            for bridge_pos in bridges:
                if bridge_pos in valid_moves:
                    return bridge_pos
        
        # 4. Build path toward goal using shortest path heuristic
        scored_moves = []
        for r, c in valid_moves:
            test_game = game.clone()
            test_game.make_move(r, c)
            
            # Shortest path improvement
            my_dist = shortest_path_distance_cached(test_game, current_player)
            opp_dist = shortest_path_distance_cached(test_game, opponent)
            
            path_score = float(opp_dist - my_dist) if my_dist < 10**7 and opp_dist < 10**7 else 0.0
            
            # Center preference (but less than path)
            center = (game.size - 1) / 2.0
            center_score = -0.1 * (abs(r - center) + abs(c - center))
            
            # Proximity to own stones (for connectivity)
            own_stones = [(sr, sc) for sr in range(game.size) for sc in range(game.size) 
                         if game.board[sr][sc] == current_player]
            if own_stones:
                min_prox = min(abs(r - sr) + abs(c - sc) for sr, sc in own_stones)
                prox_score = -0.05 * min_prox
            else:
                prox_score = 0.0
            
            total_score = path_score + center_score + prox_score
            scored_moves.append((total_score, (r, c)))
        
        scored_moves.sort(reverse=True, key=lambda x: x[0])
        return scored_moves[0][1] if scored_moves else random.choice(valid_moves)
    
    def decide_swap(self, game: HexGame):
        """Intermediate bot evaluates swap based on center position"""
        if not (len(game.move_history) == 1 and game.swap_offered):
            return False
        
        # Check if first move is in center (strong opening)
        if game.move_history:
            first_move = game.move_history[0][1]
            center = (game.size - 1) / 2.0
            dist_from_center = abs(first_move[0] - center) + abs(first_move[1] - center)
            
            # If very close to center, swap (center is strong)
            if dist_from_center < 2.0:
                return True
        
        return False

class ExpertBot:
    """Expert Bot - Uses Neural MCTS (Grandmaster level)"""
    def __init__(self, net, name="Expert Bot", simulations=MCTS_SIMS):
        self.name = name
        self.mcts_agent = NeuralMCTSAgent(net, name=name, simulations=simulations, cpuct=1.3)
    
    def select_move(self, game: HexGame):
        return self.mcts_agent.select_move(game)
    
    def decide_swap(self, game: HexGame):
        return self.mcts_agent.decide_swap(game)

# MCTS data structure
class MCTSNode:
    def __init__(self, game, parent=None, move=None, prior=0.0):
        self.game=game; self.parent=parent; self.move=move; self.prior=prior
        self.children={}; self.visits=0; self.value_sum=0.0
    def q_value(self): return (self.value_sum/self.visits) if self.visits>0 else 0.0

# -------------------- PyTorch Enhanced Net --------------------
class TinyNet(nn.Module):
    def __init__(self, board_size=BOARD_SIZE, hidden=256, lr=1e-3, device="cpu"):
        super().__init__()
        if not TORCH_AVAILABLE: raise ImportError("PyTorch required")
        self.board_size=board_size; self.n_cells=board_size*board_size; self.hidden=hidden; self.device=device
        # Enhanced architecture with residual connections
        self.input_layer=nn.Linear(2*self.n_cells, hidden)
        self.hidden_layer1=nn.Linear(hidden, hidden)
        self.hidden_layer2=nn.Linear(hidden, hidden)
        self.policy_head=nn.Linear(hidden, self.n_cells)
        self.value_head=nn.Linear(hidden, 1)
        self.optimizer=optim.Adam(self.parameters(), lr=lr)
        self.value_loss_fn=nn.MSELoss()
        self.to(self.device)
    def board_to_input(self, game: HexGame, perspective):
        N=self.n_cells
        p=torch.zeros(N,dtype=torch.float32); o=torch.zeros(N,dtype=torch.float32)
        idx=0
        for r in range(game.size):
            for c in range(game.size):
                v=game.board[r][c]
                if v==perspective: p[idx]=1.0
                elif v!=HexGame.EMPTY: o[idx]=1.0
                idx+=1
        return torch.cat([p,o])
    def forward(self,x):
        h=torch.tanh(self.input_layer(x))
        h1=torch.tanh(self.hidden_layer1(h))
        h2=torch.tanh(self.hidden_layer2(h1+h))  # Residual connection
        logits=self.policy_head(h2)
        value=torch.tanh(self.value_head(h2)).reshape(-1)
        return logits,value
    def policy_value(self, game: HexGame, perspective):
        self.eval()
        x=self.board_to_input(game,perspective).unsqueeze(0).to(self.device)
        with torch.no_grad(): logits,value=self.forward(x)
        logits=logits.squeeze(0)
        mask=torch.zeros(self.n_cells,dtype=torch.float32,device=self.device)
        val_idx=[r*self.board_size+c for r,c in game.valid_moves()]
        if val_idx: mask[val_idx]=1.0
        else: mask.fill_(1.0/self.n_cells)
        masked=logits.masked_fill(mask==0,-float('inf'))
        probs=F.softmax(masked,dim=-1)
        return probs.cpu().numpy(), float(value.item())
    def train_batch(self, X, P, V, epochs=1):
        self.train()
        x=torch.from_numpy(X).float().to(self.device)
        p=torch.from_numpy(P).float().to(self.device)
        v=torch.from_numpy(V).float().to(self.device)
        for _ in range(epochs):
            self.optimizer.zero_grad()
            logits, v_pred = self.forward(x)
            log_probs = torch.log_softmax(logits, dim=-1)
            policy_loss = -(p*log_probs).sum(dim=1).mean()
            value_loss = self.value_loss_fn(v_pred, v)
            (policy_loss+value_loss).backward(); self.optimizer.step()
    def save_weights(self, path):
        torch.save(self.state_dict(), path); print(f"[SAVE] Weights -> {path}")
    def load_weights(self, path):
        try:
            state_dict = torch.load(path, map_location=self.device)
            # Handle compatibility with old model architecture
            model_keys = set(self.state_dict().keys())
            loaded_keys = set(state_dict.keys())
            
            # If old architecture (no hidden layers), create compatible state
            if 'hidden_layer1.weight' not in loaded_keys and 'hidden_layer1.weight' in model_keys:
                print("[LOAD] Old architecture detected, adapting weights...")
                # Copy existing weights and initialize new layers
                new_state = self.state_dict()
                for key in loaded_keys:
                    if key in new_state:
                        new_state[key] = state_dict[key]
                # Initialize new layers with small random values
                for key in model_keys:
                    if key not in loaded_keys and 'hidden_layer' in key:
                        if 'weight' in key:
                            torch.nn.init.xavier_uniform_(new_state[key])
                        else:
                            torch.nn.init.zeros_(new_state[key])
                state_dict = new_state
            
            self.load_state_dict(state_dict, strict=False)
            self.to(self.device)
            print(f"[LOAD] Weights <- {path}")
        except FileNotFoundError:
            print(f"[LOAD] No weights at {path}; using random init")
        except Exception as e:
            print(f"[LOAD] Error loading weights: {e}; using random init")

# -------------------- Neural MCTS (strong) --------------------
class NeuralMCTSAgent:
    """Enhanced Neural MCTS agent with expert-level evaluation and optimizations"""
    def __init__(self, net: TinyNet, name="NeuralMCTS", simulations=MCTS_SIMS, cpuct=1.3, use_mp=USE_MP):
        self.name = name
        self.net = net
        self.simulations = simulations
        self.cpuct = cpuct
        self.use_mp = use_mp
        self.dirichlet_alpha = 0.3  # For exploration
        self.dirichlet_epsilon = 0.25  # For exploration
        self.virtual_loss = 3.0  # For parallel simulations
        
    def _immediate_win(self, game, player):
        """Check for immediate winning move"""
        for mv in game.valid_moves():
            g = game.clone()
            g.make_move(*mv)
            if g.terminal and g.winner == player:
                return mv
        return None
    
    def _opponent_threats(self, game, opponent):
        """Find opponent threats that must be blocked - including multi-move threats"""
        threats = []
        
        # Check immediate wins (1-move threats)
        for mv in game.valid_moves():
            g = game.clone()
            if g.turn != opponent:
                g.turn = opponent
            if g.is_valid_move(*mv):
                g.make_move(*mv)
                if g.terminal and g.winner == opponent:
                    threats.append((mv, 1))  # Priority 1: immediate win
        
        # Check 2-move threats (opponent can win in 2 moves)
        if not threats:  # Only check if no immediate threats
            for mv in game.valid_moves():
                g = game.clone()
                if g.turn != opponent:
                    g.turn = opponent
                if g.is_valid_move(*mv):
                    g.make_move(*mv)
                    # Check if opponent can win next move
                    for mv2 in g.valid_moves():
                        g2 = g.clone()
                        g2.make_move(*mv2)
                        if g2.terminal and g2.winner == opponent:
                            threats.append((mv, 2))  # Priority 2: can win in 2 moves
                            break
        
        # Check critical path blocking (opponent's shortest path is getting too short)
        if not threats:
            opp_dist = shortest_path_distance_cached(game, opponent)
            if opp_dist < 3:  # Opponent is very close to winning
                # Find moves that significantly increase opponent's distance
                best_block = None
                best_improvement = -1
                for mv in game.valid_moves():
                    g = game.clone()
                    g.make_move(*mv)
                    new_opp_dist = shortest_path_distance_cached(g, opponent)
                    improvement = opp_dist - new_opp_dist
                    if improvement < best_improvement:  # More negative = better blocking
                        best_improvement = improvement
                        best_block = mv
                if best_block:
                    threats.append((best_block, 3))  # Priority 3: critical path block
        
        return [mv for mv, _ in threats]  # Return just the moves
    
    def _safe_moves(self, game):
        """Find moves that don't allow immediate opponent win"""
        me = game.turn
        opp = HexGame.P1 if me == HexGame.P2 else HexGame.P2
        safe = []
        for mv in game.valid_moves():
            g = game.clone()
            g.make_move(*mv)
            if self._immediate_win(g, opp) is None:
                safe.append(mv)
        return set(safe)
    
    def _blend_root_priors(self, game, priors, alpha=0.9, lam=0.9):
        """Blend neural network priors with fast heuristics (optimized for speed)"""
        size = game.size
        vm = game.valid_moves()
        net_map = {}
        for idx, p in enumerate(priors):
            r = idx // size
            c = idx % size
            if (r, c) in vm:
                net_map[(r, c)] = float(p)
        
        # Use FAST heuristic instead of expensive expert evaluation
        # Simple center-distance heuristic (much faster)
        heu_map = {}
        center = (game.size - 1) / 2.0
        me = game.turn
        
        # Fast heuristic: prefer center and existing stones
        for mv in vm:
            r, c = mv
            # Center preference (fast calculation)
            center_dist = abs(r - center) + abs(c - center)
            center_score = 1.0 / (1.0 + center_dist * 0.1)
            
            # Proximity to own stones (fast)
            prox_score = 0.0
            for nr, nc in game.neighbors(r, c):
                if game.board[nr][nc] == me:
                    prox_score += 0.5
            
            # Combine into simple heuristic
            heu = center_score + prox_score * 0.3
            heu_map[mv] = heu
        
        s_net = sum(net_map.values()) or 1.0
        s_heu = sum(heu_map.values()) or 1.0
        blend = {}
        for mv in vm:
            pn = net_map.get(mv, 1e-8) / s_net
            ph = heu_map.get(mv, 0.0) / s_heu
            # Use mostly neural network (90%) with fast heuristic (10%)
            blend[mv] = alpha * pn + (1.0 - alpha) * ph
        
        s = sum(blend.values()) or 1.0
        for k in blend:
            blend[k] /= s
        return blend
    
    def _endgame_pick(self, game):
        """Use AlphaBeta for precise endgame play"""
        empties = len(game.valid_moves())
        if empties > 30:  # Only use in true endgame
            return None
        # Use deeper search for endgame - critical for winning
        ab = AlphaBetaAgent(name="AB-End", depth=8, rollout_evals=10, consider_k=100)
        return ab.select_move(game)
    
    def decide_swap(self, game: HexGame):
        """Enhanced swap decision using expert evaluation"""
        if not (len(game.move_history) == 1 and game.swap_offered):
            return False
        
        # P2 perspective value: higher is better for P2
        gA = game.clone()  # no swap
        gB = game.clone()
        gB.do_swap()  # swapped colors
        
        # Use expert evaluation for more accurate swap decision
        eval_A = expert_eval_cached(gA, HexGame.P2)
        eval_B = expert_eval_cached(gB, HexGame.P2)
        
        # Also consider network evaluation
        _, vA = self.net.policy_value(gA, HexGame.P2)
        _, vB = self.net.policy_value(gB, HexGame.P2)
        
        # Blend expert and network evaluations
        combined_A = 0.7 * eval_A + 0.3 * vA * 1000.0
        combined_B = 0.7 * eval_B + 0.3 * vB * 1000.0
        
        return combined_B > combined_A
    
    def select_move(self, game: HexGame, return_policy_target=False):
        """Enhanced MCTS with expert evaluation integration"""
        # Safety: Check if game is already terminal
        if game.terminal:
            return None if not return_policy_target else (None, np.zeros(self.net.n_cells, dtype=np.float32))
        
        # Safety: Check if there are valid moves
        vm_check = game.valid_moves()
        if not vm_check:
            return None if not return_policy_target else (None, np.zeros(self.net.n_cells, dtype=np.float32))
        
        me = game.turn
        opp = HexGame.P1 if me == HexGame.P2 else HexGame.P2
        
        # Immediate win
        mv = self._immediate_win(game, me)
        if mv is not None and not return_policy_target:
            return mv
        
        # Immediate block - CRITICAL: Always block threats
        threats = self._opponent_threats(game, opp)
        if threats and not return_policy_target:
            # Always block the first threat (highest priority)
            # For immediate wins, block immediately
            for t in threats:
                g = game.clone()
                if g.turn != opp:
                    g.turn = opp
                if g.is_valid_move(*t):
                    g.make_move(*t)
                    if g.terminal and g.winner == opp:
                        # This is an immediate win threat - block it!
                        return t
            
            # For other threats, pick the best blocking move
            best = None
            best_score = -1e18
            for t in threats:
                g = game.clone()
                g.make_move(*t)
                # Evaluate how good this blocking move is
                # Prioritize moves that block AND improve our position
                my_dist_after = shortest_path_distance_cached(g, me)
                opp_dist_after = shortest_path_distance_cached(g, opp)
                
                # Score: lower opponent distance is better, higher our distance improvement is better
                if my_dist_after < 10**7 and opp_dist_after < 10**7:
                    score = float(opp_dist_after - my_dist_after) * 100.0  # Strong blocking bonus
                else:
                    score = -expert_eval_cached(g, me)  # Fallback to expert eval
                
                if score > best_score:
                    best_score = score
                    best = t
            
            if best:
                return best
        
        # Endgame deeper search
        ab_mv = self._endgame_pick(game)
        if ab_mv is not None and not return_policy_target:
            return ab_mv
        
        # Root setup
        root = MCTSNode(game.clone(), parent=None, move=None, prior=1.0)
        priors, _ = self.net.policy_value(game, me)
        vm = game.valid_moves()
        if not vm:
            return None if not return_policy_target else (None, np.zeros(self.net.n_cells, dtype=np.float32))
        
        safe = self._safe_moves(game)
        blended = self._blend_root_priors(game, priors, alpha=0.65, lam=0.9)
        
        # Add Dirichlet noise for exploration
        if len(vm) > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(vm))
            for i, mv2 in enumerate(vm):
                blended[mv2] = (1 - self.dirichlet_epsilon) * blended.get(mv2, 1e-8) + self.dirichlet_epsilon * noise[i]
        
        # Boost safe moves significantly (critical for blocking)
        for mv2 in vm:
            if mv2 in safe:
                blended[mv2] = blended.get(mv2, 1e-8) * 2.0  # Strong boost for safe moves
        
        # Additional: prioritize blocking opponent's path if they're close to winning
        # BUT also prioritize building our own path - balanced approach
        opp_dist = shortest_path_distance_cached(game, opp)
        my_dist = shortest_path_distance_cached(game, me)
        
        # Balanced strategy: block when behind, build when ahead
        if opp_dist < 10**7 and my_dist < 10**7:
            if opp_dist < my_dist + 2:  # Opponent is close or ahead - prioritize blocking
                # Boost moves that block opponent's path
                for mv2 in vm:
                    g_test = game.clone()
                    g_test.make_move(*mv2)
                    new_opp_dist = shortest_path_distance_cached(g_test, opp)
                    new_my_dist = shortest_path_distance_cached(g_test, me)
                    if new_opp_dist > opp_dist:  # This move increases opponent's distance
                        blocking_bonus = (new_opp_dist - opp_dist) * 3.0  # Reduced from 5.0 for balance
                        # Also reward moves that improve our position
                        if new_my_dist < my_dist:
                            building_bonus = (my_dist - new_my_dist) * 2.0
                            blended[mv2] = blended.get(mv2, 1e-8) * (1.0 + blocking_bonus + building_bonus)
                        else:
                            blended[mv2] = blended.get(mv2, 1e-8) * (1.0 + blocking_bonus)
            else:  # We're ahead - prioritize building our path
                # Boost moves that improve our path
                for mv2 in vm:
                    g_test = game.clone()
                    g_test.make_move(*mv2)
                    new_my_dist = shortest_path_distance_cached(g_test, me)
                    if new_my_dist < my_dist:  # This move improves our path
                        building_bonus = (my_dist - new_my_dist) * 4.0
                        blended[mv2] = blended.get(mv2, 1e-8) * (1.0 + building_bonus)
        
        s = sum(blended.get(mv2, 0.0) for mv2 in vm) or 1.0
        for mv2 in vm:
            pr = blended.get(mv2, 1e-8) / s
            cg = game.clone()
            cg.make_move(*mv2)
            root.children[mv2] = MCTSNode(cg, parent=root, move=mv2, prior=pr)
        
        # Enhanced simulations with expert evaluation and early stopping
        best_visits = 0
        max_selection_depth = 50000  # Reduced to prevent deep loops
        sim_timeout = 0.1  # Max time per simulation (100ms)
        total_timeout = 1000.0  # Max total time for all simulations (10 seconds)
        sim_start_time = time.time()
        
        for sim in range(self.simulations):
            # Safety: Check total time
            if time.time() - sim_start_time > total_timeout:
                print(f"[MCTS] WARNING: Total timeout reached at sim {sim}/{self.simulations}, breaking")
                break
            
            node = root
            selection_depth = 0
            sim_iter_start = time.time()
            
            # Selection phase with PUCT
            while node.children and selection_depth < max_selection_depth:
                # Safety: Check per-simulation time
                if time.time() - sim_iter_start > sim_timeout:
                    break
                
                selection_depth += 1
                # Add virtual loss for parallel simulations
                best_score = -1e18
                best_child = None
                for child in node.children.values():
                    puct_score = child.q_value() + self.cpuct * child.prior * math.sqrt(max(1, node.visits)) / (1 + child.visits)
                    if puct_score > best_score:
                        best_score = puct_score
                        best_child = child
                if best_child is None:
                    break  # No valid child found
                node = best_child
            
            if selection_depth >= max_selection_depth:
                print(f"[MCTS] WARNING: Max selection depth reached at sim {sim}, breaking")
                break
            
            g = node.game
            
            if g.terminal:
                self._bp_terminal(node, g.winner)
                continue
            
            persp = g.turn
            try:
                # Safety: Add timeout for network evaluation
                eval_start = time.time()
                p, v = self.net.policy_value(g, persp)
                eval_time = time.time() - eval_start
                if eval_time > 1.0:  # If evaluation takes >1s, something's wrong
                    print(f"[MCTS] WARNING: Network eval took {eval_time:.2f}s")
            except Exception as e:
                print(f"[MCTS] ERROR: Network evaluation failed: {e}, using default value")
                v = 0.0
                p = np.ones(self.net.n_cells, dtype=np.float32) / self.net.n_cells
            
            vm2 = g.valid_moves()
            
            if not vm2:
                self._bp_value(node, v)
                continue
            
            # Expansion phase
            mp_map = {}
            for idx, pr in enumerate(p):
                r = idx // g.size
                c = idx % g.size
                if (r, c) in vm2:
                    mp_map[(r, c)] = float(pr)
            
            s2 = sum(mp_map.values()) or 1.0
            for m in vm2:
                if m not in node.children:
                    pr = mp_map.get(m, 1e-8) / s2
                    cg = g.clone()
                    cg.make_move(*m)
                    node.children[m] = MCTSNode(cg, parent=node, move=m, prior=pr)
            
            # Use neural network value with occasional expert evaluation for better play
            # Blend expert evaluation every 10th simulation for stronger play
            if sim % 10 == 0 and sim > 50:  # Only after some exploration
                try:
                    expert_val = expert_eval_cached(g, persp) / 1000.0
                    blended_val = 0.7 * v + 0.3 * expert_val
                    self._bp_value(node, blended_val)
                except:
                    self._bp_value(node, v)
            else:
                self._bp_value(node, v)
            
            # Early stopping if one move is clearly dominant (less aggressive for better play)
            if sim > 50 and sim % 20 == 0:  # Check less frequently to allow more exploration
                child_visits = [child.visits for child in root.children.values()]
                if child_visits:
                    max_visits = max(child_visits)
                    second_max = sorted(child_visits)[-2] if len(child_visits) > 1 else 0
                    # Only stop early if move is VERY dominant (less aggressive)
                    if max_visits > second_max * 4 and max_visits > self.simulations // 2:
                        # One move is clearly dominant, stop early
                        break
        
        # Safely get best move - ensure we have valid children
        if not root.children:
            # Fallback: return first valid move
            vm = game.valid_moves()
            if vm:
                best_move = vm[0]
            else:
                return None if not return_policy_target else (None, np.zeros(self.net.n_cells, dtype=np.float32))
        else:
            try:
                best_move = max(root.children.items(), key=lambda kv: kv[1].visits)[0]
            except Exception as e:
                print(f"[MCTS] ERROR: Failed to select best move: {e}, using first valid move")
                vm = game.valid_moves()
                best_move = vm[0] if vm else None
        
        if return_policy_target:
            pi = np.zeros(self.net.n_cells, dtype=np.float32)
            tot = sum(c.visits for c in root.children.values())
            if tot > 0:
                for (r, c), child in root.children.items():
                    pi[r * game.size + c] = child.visits / tot
            return best_move, pi
        
        return best_move
    
    def _bp_terminal(self, node, winner):
        """Backpropagate terminal result"""
        cur = node
        while cur is not None:
            cur.visits += 1
            if cur.game.move_history:
                mover = cur.game.move_history[-1][0]
                if winner == mover:
                    cur.value_sum += 1.0
            cur = cur.parent
    
    def _bp_value(self, node, value):
        """Backpropagate value"""
        cur = node
        v = (value + 1.0) / 2.0  # Normalize to [0, 1]
        while cur is not None:
            cur.visits += 1
            cur.value_sum += v
            v = 1.0 - v  # Alternate perspective
            cur = cur.parent

# -------------------- Self-play training --------------------
def format_eta(seconds):
    if seconds is None: return "N/A"
    h=int(seconds//3600); m=int((seconds%3600)//60); s=int(seconds%60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def self_play_episode(net: TinyNet, mcts: NeuralMCTSAgent, temp=1.0, screen=None, cell_size=0, margin=0, openings_seen=None):
    game=HexGame(size=net.board_size, swap_rule=True)
    states=[]; policies=[]; players=[]
    max_moves = net.board_size * net.board_size  # Maximum possible moves
    move_count = 0
    move_timeout = 200.0  # Maximum time per move (30 seconds)
    episode_start_time = time.time()
    episode_timeout = 2000.0  # Maximum time per episode (5 minutes)
    
    # Safety: Check initial state
    if not game.valid_moves():
        print(f"[TRAIN] WARNING: No valid moves at start, skipping game")
        return [], game
    
    consecutive_failures = 0
    max_consecutive_failures = 5
    
    while not game.terminal and game.valid_moves():
        # Safety checks to prevent infinite loops
        move_count += 1
        if move_count > max_moves:
            print(f"[TRAIN] WARNING: Exceeded max moves ({max_moves}), forcing termination")
            # Determine winner by evaluation
            try:
                eval_p1 = expert_eval_cached(game, HexGame.P1)
                eval_p2 = expert_eval_cached(game, HexGame.P2)
                game.winner = HexGame.P1 if eval_p1 > eval_p2 else HexGame.P2
            except:
                game.winner = HexGame.P1  # Default winner
            game.terminal = True
            break
        
        if time.time() - episode_start_time > episode_timeout:
            print(f"[TRAIN] WARNING: Episode timeout ({episode_timeout}s), forcing termination")
            try:
                eval_p1 = expert_eval_cached(game, HexGame.P1)
                eval_p2 = expert_eval_cached(game, HexGame.P2)
                game.winner = HexGame.P1 if eval_p1 > eval_p2 else HexGame.P2
            except:
                game.winner = HexGame.P1
            game.terminal = True
            break
        
        # Safety: Check if game state is valid
        if game.terminal:
            break
        
        valid_check = game.valid_moves()
        if not valid_check:
            print(f"[TRAIN] WARNING: No valid moves at move {move_count}, forcing termination")
            try:
                eval_p1 = expert_eval_cached(game, HexGame.P1)
                eval_p2 = expert_eval_cached(game, HexGame.P2)
                game.winner = HexGame.P1 if eval_p1 > eval_p2 else HexGame.P2
            except:
                game.winner = HexGame.P1
            game.terminal = True
            break
        
        # Get move with timeout protection
        move_start_time = time.time()
        mv = None
        pi = None
        try:
            mv,pi=mcts.select_move(game, return_policy_target=True)
            move_time = time.time() - move_start_time
            if move_time > move_timeout:
                print(f"[TRAIN] WARNING: Move took {move_time:.1f}s (timeout: {move_timeout}s)")
            consecutive_failures = 0  # Reset on success
        except Exception as e:
            print(f"[TRAIN] ERROR: Exception in select_move: {e}")
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                print(f"[TRAIN] ERROR: Too many consecutive failures ({consecutive_failures}), forcing termination")
                try:
                    eval_p1 = expert_eval_cached(game, HexGame.P1)
                    eval_p2 = expert_eval_cached(game, HexGame.P2)
                    game.winner = HexGame.P1 if eval_p1 > eval_p2 else HexGame.P2
                except:
                    game.winner = HexGame.P1
                game.terminal = True
                break
            # Fallback to random move
            valid = game.valid_moves()
            if valid:
                mv = random.choice(valid)
                pi = np.zeros(net.n_cells, dtype=np.float32)
                idx = mv[0] * game.size + mv[1]
                if idx < len(pi):
                    pi[idx] = 1.0
            else:
                break
        
        if mv is None: 
            print(f"[TRAIN] WARNING: No move returned at move {move_count}, breaking")
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                print(f"[TRAIN] ERROR: Too many consecutive failures, forcing termination")
                try:
                    eval_p1 = expert_eval_cached(game, HexGame.P1)
                    eval_p2 = expert_eval_cached(game, HexGame.P2)
                    game.winner = HexGame.P1 if eval_p1 > eval_p2 else HexGame.P2
                except:
                    game.winner = HexGame.P1
                game.terminal = True
                break
            continue
        
        current_player = game.turn
        opponent = HexGame.P1 if current_player == HexGame.P2 else HexGame.P2
        
        # First, check if we can win in one move (highest priority)
        our_win_move = None
        for test_mv in game.valid_moves():
            test_game = game.clone()
            if test_game.is_valid_move(*test_mv):
                test_game.make_move(*test_mv)
                if test_game.terminal and test_game.winner == current_player:
                    our_win_move = test_mv
                    break
        
        # If we can win, do it
        if our_win_move:
            blocking_idx = our_win_move[0] * game.size + our_win_move[1]
            pi = np.zeros_like(pi)
            if blocking_idx < len(pi):
                pi[blocking_idx] = 1.0
            mv = our_win_move
        else:
            # Block opponent win if they can win in one move
            opponent_win_moves = []
            for test_mv in game.valid_moves():
                test_game = game.clone()
                test_game.turn = opponent  # Temporarily set to opponent's turn
                if test_game.is_valid_move(*test_mv):
                    test_game.make_move(*test_mv)
                    if test_game.terminal and test_game.winner == opponent:
                        opponent_win_moves.append(test_mv)
            
            # If opponent can win, block it (use first blocking move found)
            if opponent_win_moves:
                blocking_move = opponent_win_moves[0]
                # Update pi to reflect the blocking move
                blocking_idx = blocking_move[0] * game.size + blocking_move[1]
                pi = np.zeros_like(pi)
                if blocking_idx < len(pi):
                    pi[blocking_idx] = 1.0
                mv = blocking_move
        
        # Diversify opening: ENSURE each game uses a different opening
        if len(game.move_history)==0 and openings_seen is not None:
            valid_moves = game.valid_moves()
            unseen = [mv2 for mv2 in valid_moves if mv2 not in openings_seen]
            
            if unseen:
                # Prefer unseen openings - use MCTS policy but only from unseen
                probs = np.array([pi[r*game.size+c] for (r,c) in unseen], dtype=np.float64)
                ps = probs.sum()
                if ps <= 0:
                    probs = np.ones(len(unseen), dtype=np.float64) / len(unseen)
                else:
                    probs = probs / ps
                choice = np.random.choice(len(unseen), p=probs)
                mv = unseen[choice]
                # Record this opening as used
                openings_seen.add(mv)
            else:
                # All openings seen in this iteration - reset and pick randomly
                # This ensures we get diverse openings across iterations
                openings_seen.clear()
                # Pick a random opening from valid moves
                mv = random.choice(valid_moves)
                openings_seen.add(mv)
        x=net.board_to_input(game, game.turn).cpu().numpy()
        states.append(x); policies.append(pi); players.append(game.turn)
        if temp!=0:
            valid=[r*game.size+c for r,c in game.valid_moves()]
            probs=pi[valid]; ps=probs.sum()
            probs=(probs/ps) if ps>0 else np.ones(len(valid))/len(valid)
            probs=probs**(1.0/temp); probs=probs/(probs.sum()+1e-12)
            idx=np.random.choice(len(valid), p=probs); linear=valid[idx]; mv=(linear//game.size, linear%game.size)
        # Validate move before making it
        if not game.is_valid_move(*mv):
            print(f"[TRAIN] WARNING: Invalid move {mv} at move {move_count}, skipping")
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                print(f"[TRAIN] ERROR: Too many invalid moves, forcing termination")
                try:
                    eval_p1 = expert_eval_cached(game, HexGame.P1)
                    eval_p2 = expert_eval_cached(game, HexGame.P2)
                    game.winner = HexGame.P1 if eval_p1 > eval_p2 else HexGame.P2
                except:
                    game.winner = HexGame.P1
                game.terminal = True
                break
            continue
        
        try:
            game.make_move(*mv)
        except Exception as e:
            print(f"[TRAIN] ERROR: Failed to make move {mv}: {e}")
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                print(f"[TRAIN] ERROR: Too many move failures, forcing termination")
                try:
                    eval_p1 = expert_eval_cached(game, HexGame.P1)
                    eval_p2 = expert_eval_cached(game, HexGame.P2)
                    game.winner = HexGame.P1 if eval_p1 > eval_p2 else HexGame.P2
                except:
                    game.winner = HexGame.P1
                game.terminal = True
                break
            continue
        
        # Safety check: ensure game state is valid
        if game.terminal:
            break
        
        # Progress indicator every 20 moves
        if move_count % 20 == 0:
            print(f"[TRAIN] Progress: Move {move_count}, {len(game.valid_moves())} moves remaining")
        
        # Optimized visualization - only update every few moves to reduce lag
        if screen:
            # Only render every 3rd move for better performance
            if len(game.move_history) % 3 == 0 or game.terminal:
                render_game(screen, game, net.board_size, cell_size, margin)
                for event in pygame.event.get():
                    if event.type==pygame.QUIT: return [], None
                time.sleep(0.01)  # Reduced from 0.03
        if game.swap_rule and len(game.move_history)==1 and not game.swap_offered:
            game.offer_swap()
            try:
                if mcts._immediate_win(game, HexGame.P2) is not None:
                    game.do_swap()
                    if screen:
                        render_game(screen, game, net.board_size, cell_size, margin)
                        for event in pygame.event.get():
                            if event.type==pygame.QUIT: return [], None
                        time.sleep(0.01)  # Reduced from 0.03
            except Exception as e:
                print(f"[TRAIN] WARNING: Error in swap logic: {e}")
    
    # Ensure game has a winner
    if not game.terminal:
        print(f"[TRAIN] WARNING: Game not terminal after {move_count} moves, forcing evaluation")
        eval_p1 = expert_eval_cached(game, HexGame.P1)
        eval_p2 = expert_eval_cached(game, HexGame.P2)
        game.winner = HexGame.P1 if eval_p1 > eval_p2 else HexGame.P2
        game.terminal = True
    
    z = 1 if game.winner==HexGame.P1 else -1
    out=[]
    for x,pi,pl in zip(states,policies,players):
        out.append((x, pi, float(z if pl==HexGame.P1 else -z)))
    return out, game


def train_selfplay(net: TinyNet, iterations=TRAIN_ITERS, games_per_iter=SELFPLAY_GAMES, visualize=False):
    global PYGAME_AVAILABLE
    screen=None; CELL=0; MARGIN=0
    # Disable visualization by default for faster training (can enable with --visualize flag)
    # Visualization causes significant lag during training
    if visualize and PYGAME_AVAILABLE:
        pygame.init(); CELL=26; MARGIN=18
        size=net.board_size; screen_size=MARGIN*2 + CELL*size + CELL//2*(size-1)
        os.environ['SDL_VIDEO_CENTERED']='1'
        try:
            screen=pygame.display.set_mode((screen_size,screen_size))
            pygame.display.set_caption("Hex Training Visualization")
        except Exception as e:
            print(f"[TRAIN] Pygame display error: {e}. Disabling visualization.")
            PYGAME_AVAILABLE=False; screen=None
    elif visualize:
        print("[TRAIN] Pygame not available.")
    # Improved MCTS for training - balanced for speed and quality
    # Reduced simulations for faster training (can increase later if needed)
    mcts=NeuralMCTSAgent(net, simulations=150, cpuct=1.5)  # Reduced from 300 for speed
    buffer=[]; loop_start=time.time()
    for it in range(iterations):
        it_start=time.time(); elapsed=time.time()-loop_start
        avg=(elapsed/(it)) if it>0 else None
        eta=avg*(iterations-(it+1)) if avg else None
        print(f"[TRAIN] Iter {it+1}/{iterations} (ETA {format_eta(eta)}) - {games_per_iter} self-play games")
        iter_examples=[]; p1w=0; p2w=0
        openings_seen=set()  # Track openings to ensure diversity
        for gk in range(games_per_iter):
            # Reset openings_seen for each game to ensure maximum diversity
            # Each game should try a different opening
            game_openings = set()
            game_start = time.time()
            game_timeout = 600.0  # 10 minutes max per game
            
            try:
                ex, gm = self_play_episode(net, mcts, temp=1.0, screen=screen, cell_size=CELL, margin=MARGIN, openings_seen=game_openings)
                
                # Check timeout
                game_elapsed = time.time() - game_start
                if game_elapsed > game_timeout:
                    print(f"[TRAIN] WARNING: Game {gk+1} exceeded timeout ({game_timeout}s), skipping")
                    continue
                    
            except Exception as e:
                print(f"[TRAIN] ERROR: Game {gk+1} failed with exception: {e}")
                import traceback
                traceback.print_exc()
                print(f"[TRAIN] Skipping this game and continuing...")
                continue
            
            if gm is None:
                print("[TRAIN] Window closed; stopping.");
                if screen: pygame.quit(); return net
            
            # Validate game result
            if not gm.terminal or gm.winner is None:
                print(f"[TRAIN] WARNING: Game {gk+1} did not terminate properly, skipping")
                continue
            
            if gm.winner==HexGame.P1: p1w+=1
            elif gm.winner==HexGame.P2: p2w+=1
            game_time = time.time() - game_start
            
            # Validate examples
            if len(ex) == 0:
                print(f"[TRAIN] WARNING: Game {gk+1} produced no examples, skipping")
                continue
            
            print(f"[TRAIN]  Game {gk+1}/{games_per_iter} -> {'P1' if gm.winner==HexGame.P1 else 'P2'} ({game_time:.1f}s, {len(ex)} positions, opening: {gm.move_history[0][1] if gm.move_history else 'N/A'})")
            iter_examples.extend(ex)
            # Add this game's opening to the iteration set
            if gm.move_history:
                openings_seen.add(gm.move_history[0][1])
        buffer.extend(iter_examples)
        if len(buffer)>60000: buffer=buffer[-60000:]
        if not buffer:
            print("[TRAIN] No examples; skip train"); continue
        print(f"[TRAIN]  Summary: P1={p1w}, P2={p2w}; Positions={len(iter_examples)} in {time.time()-it_start:.1f}s; Buffer={len(buffer)}. Training...")
        # Use larger sample size for better learning
        sample_size = min(len(buffer), 8192)  # Increased from 4096
        sample = random.sample(buffer, sample_size)
        X=np.vstack([e[0] for e in sample]); P=np.vstack([e[1] for e in sample]); V=np.array([e[2] for e in sample], dtype=np.float32)
        # Train for more epochs to learn better
        net.train_batch(X,P,V,epochs=3)  # Increased from 2
        print(f"[TRAIN]  Iter {it+1} done. Trained on {sample_size} positions.")
    print(f"[TRAIN] Finished in {format_eta(time.time()-loop_start)}")
    if screen: pygame.quit()
    return net

# -------------------- Analysis Engine --------------------
class AnalysisEngine:
    """Lightweight analysis engine for move evaluation"""
    def __init__(self, depth=1):  # Reduced default depth for better performance
        self.depth = depth
        self.cache = {}
        self.cache_max_size = 100
    
    def _minimax(self, game: HexGame, depth: int, alpha: float, beta: float, 
                 perspective: int, maximizing: bool):
        """Minimax with alpha-beta pruning"""
        if game.terminal:
            if game.winner == perspective:
                return 1e6
            elif game.winner:
                return -1e6
            return 0.0
        
        if depth == 0:
            # Use expert evaluation at leaf nodes
            return expert_eval(game, perspective)
        
        valid_moves = game.valid_moves()
        if not valid_moves:
            return 0.0
        
        # Check for immediate wins/blocks first
        current_player = game.turn
        opponent = HexGame.P1 if current_player == HexGame.P2 else HexGame.P2
        
        # Immediate win check
        if current_player == perspective:
            for mv in valid_moves:
                test_game = game.clone()
                test_game.make_move(*mv)
                if test_game.terminal and test_game.winner == perspective:
                    return 1e6 - (self.depth - depth)  # Prefer faster wins
        
        # Immediate block check
        if opponent == perspective:
            for mv in valid_moves:
                test_game = game.clone()
                test_game.turn = opponent
                if test_game.is_valid_move(*mv):
                    test_game.make_move(*mv)
                    if test_game.terminal and test_game.winner == opponent:
                        # Must block
                        test_game2 = game.clone()
                        test_game2.make_move(*mv)
                        return -self._minimax(test_game2, depth-1, -beta, -alpha, perspective, not maximizing)
        
        # Limit candidate moves for performance
        candidate_moves = self._get_candidate_moves(game, valid_moves, k=min(30, len(valid_moves)))
        
        if maximizing:
            max_eval = -1e18
            for mv in candidate_moves:
                test_game = game.clone()
                test_game.make_move(*mv)
                eval_score = self._minimax(test_game, depth-1, alpha, beta, perspective, False)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            return max_eval
        else:
            min_eval = 1e18
            for mv in candidate_moves:
                test_game = game.clone()
                test_game.make_move(*mv)
                eval_score = self._minimax(test_game, depth-1, alpha, beta, perspective, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            return min_eval
    
    def _get_candidate_moves(self, game: HexGame, valid_moves, k=30):
        """Get top candidate moves using heuristics"""
        if len(valid_moves) <= k:
            return valid_moves
        
        scored = []
        current_player = game.turn
        center = (game.size - 1) / 2.0
        
        for r, c in valid_moves:
            test_game = game.clone()
            test_game.make_move(r, c)
            
            # Shortest path improvement
            my_dist = shortest_path_distance_cached(test_game, current_player)
            opp = HexGame.P1 if current_player == HexGame.P2 else HexGame.P2
            opp_dist = shortest_path_distance_cached(test_game, opp)
            
            if my_dist < 10**7 and opp_dist < 10**7:
                path_score = float(opp_dist - my_dist)
            else:
                path_score = 0.0
            
            # Center preference
            center_score = -0.1 * (abs(r - center) + abs(c - center))
            
            # Bridge detection
            bridges = detect_bridges(test_game, current_player)
            bridge_score = len(bridges) * 10.0
            
            total_score = path_score + center_score + bridge_score
            scored.append((total_score, (r, c)))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        return [mv for _, mv in scored[:k]]
    
    def evaluate_position(self, game: HexGame, perspective):
        """Evaluate position using minimax search"""
        if game.terminal:
            if game.winner == perspective:
                return 1.0
            elif game.winner:
                return -1.0
            return 0.0
        
        # Use minimax for deeper evaluation
        score = self._minimax(game, self.depth, -1e18, 1e18, perspective, True)
        # Normalize to [-1, 1] range
        return max(-1.0, min(1.0, score / 1000.0))
    
    def evaluate_move(self, game: HexGame, move, perspective):
        """Evaluate a specific move using minimax"""
        g = game.clone()
        g.make_move(*move)
        
        # Evaluate the resulting position
        opponent = HexGame.P1 if perspective == HexGame.P2 else HexGame.P2
        value_after = -self._minimax(g, self.depth-1, -1e18, 1e18, opponent, True)
        
        # Normalize
        move_value = max(-1.0, min(1.0, value_after / 1000.0))
        
        # Convert to rating
        rating = move_value * 1000
        
        if abs(rating) < 10:
            rating_str = f"{rating:+.1f}"
        else:
            rating_str = f"{rating:+.0f}"
        
        return move_value, rating_str
    
    def get_best_move_and_alternatives(self, game: HexGame, top_k=5):
        """Get best move using minimax search"""
        perspective = game.turn
        valid_moves = game.valid_moves()
        
        if not valid_moves:
            return None, []
        
        # Check for immediate win first
        for mv in valid_moves:
            test_game = game.clone()
            test_game.make_move(*mv)
            if test_game.terminal and test_game.winner == perspective:
                return mv, [(mv, 1e6, "Win!")]
        
        # Check for immediate block
        opponent = HexGame.P1 if perspective == HexGame.P2 else HexGame.P2
        opponent_threats = []
        for mv in valid_moves:
            test_game = game.clone()
            test_game.turn = opponent
            if test_game.is_valid_move(*mv):
                test_game.make_move(*mv)
                if test_game.terminal and test_game.winner == opponent:
                    opponent_threats.append(mv)
        
        if opponent_threats:
            # Block the threat
            blocking_move = opponent_threats[0]
            return blocking_move, [(blocking_move, 500, "Block")]
        
        # Evaluate all moves with minimax
        move_scores = []
        candidate_moves = self._get_candidate_moves(game, valid_moves, k=min(20, len(valid_moves)))
        
        for mv in candidate_moves:
            value, rating_str = self.evaluate_move(game, mv, perspective)
            move_scores.append((mv, value, rating_str))
        
        # Sort by value (descending)
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        best_move = move_scores[0][0] if move_scores else None
        alternatives = move_scores[:top_k]
        
        return best_move, alternatives

class MoveEvaluator:
    """Evaluates moves and provides ratings - optimized for performance"""
    def __init__(self, agent):
        self.agent = agent
        # Create analysis engine with lighter depth for UI responsiveness
        # Use depth 2 for real-time, depth 4 for deep analysis (when requested)
        self.engine = AnalysisEngine(depth=2)
        self.deep_engine = AnalysisEngine(depth=4)  # For deep analysis on demand
        self.eval_cache = {}  # Cache evaluation results
        self.cache_max_size = 50
    
    def evaluate_position(self, game: HexGame, perspective, use_deep=False):
        """Evaluate position from perspective, returns value in [-1, 1]"""
        # Create cache key
        cache_key = (game.board_hash(), perspective, use_deep)
        if cache_key in self.eval_cache:
            return self.eval_cache[cache_key]
            
        if game.terminal:
            result = 0.0
            if game.winner == perspective:
                result = 1.0
            elif game.winner:
                result = -1.0
        elif isinstance(self.agent, NeuralMCTSAgent):
            _, value = self.agent.net.policy_value(game, perspective)
            result = value
        else:
            # Use ultra-fast evaluation for UI responsiveness
            if use_deep:
                result = self.deep_engine.evaluate_position(game, perspective)
            else:
                # Ultra-fast: only shortest path (cached, instant)
                my_dist = shortest_path_distance_cached(game, perspective)
                opp = HexGame.P1 if perspective == HexGame.P2 else HexGame.P2
                opp_dist = shortest_path_distance_cached(game, opp)
                
                if my_dist >= 10**7 and opp_dist >= 10**7:
                    result = 0.0
                else:
                    # Normalize to [-1, 1] range
                    score = float(opp_dist - my_dist)
                    result = max(-1.0, min(1.0, score / 20.0))
        
        # Cache the result
        if len(self.eval_cache) >= self.cache_max_size:
            # Remove oldest entries
            keys_to_remove = list(self.eval_cache.keys())[:20]
            for key in keys_to_remove:
                del self.eval_cache[key]
        self.eval_cache[cache_key] = result
        
        return result
    
    def get_best_move(self, game: HexGame, use_deep=False):
        """Get the best move for the current player - ultra fast version"""
        if game.terminal:
            return None
        
        valid_moves = game.valid_moves()
        if not valid_moves:
            return None
        
        current_player = game.turn
        
        # Check for immediate win first
        for move in valid_moves:
            g = game.clone()
            g.make_move(*move)
            if g.terminal and g.winner == current_player:
                return move
        
        # Check for immediate block (only if opponent could win next)
        opponent = HexGame.P1 if current_player == HexGame.P2 else HexGame.P2
        for move in valid_moves:
            g = game.clone()
            g.turn = opponent
            if g.is_valid_move(*move):
                g.make_move(*move)
                if g.terminal and g.winner == opponent:
                    return move  # Block the threat
        
        # For UI responsiveness, only check top 10 moves with simple heuristic
        # Pre-filter by center proximity and distance
        center = game.size // 2
        scored_moves = []
        for move in valid_moves:
            r, c = move
            # Simple heuristic: prefer center and existing stones
            score = 0
            score -= abs(r - center) + abs(c - center)  # Center preference
            
            # Check proximity to existing stones
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < game.size and 0 <= nc < game.size:
                        if game.board[nr][nc] != HexGame.EMPTY:
                            score += 2  # Prefer playing near existing stones
            
            scored_moves.append((score, move))
        
        # Sort by simple heuristic and only evaluate top 10
        scored_moves.sort(reverse=True)
        top_moves = scored_moves[:10]
        
        best_move = None
        best_value = -1e18
        
        # Evaluate only the top candidates
        for _, move in top_moves:
            g = game.clone()
            g.make_move(*move)
            # Use very light evaluation
            value = -self.evaluate_position(g, opponent, use_deep=False)
            
            if value > best_value:
                best_value = value
                best_move = move
        
        return best_move
    
    def evaluate_move(self, game: HexGame, move, perspective):
        """Evaluate a specific move, returns (value, rating_string)"""
        if isinstance(self.agent, NeuralMCTSAgent):
            # For Neural MCTS, use simpler evaluation
            g = game.clone()
            value_before = self.evaluate_position(game, perspective)
            g.make_move(*move)
            opponent = HexGame.P1 if perspective == HexGame.P2 else HexGame.P2
            value_after_opp = self.evaluate_position(g, opponent)
            value_after = -value_after_opp
            move_value = value_after - value_before
            rating = move_value * 1000
            if abs(rating) < 10:
                rating_str = f"{rating:+.1f}"
            else:
                rating_str = f"{rating:+.0f}"
            return move_value, rating_str
        else:
            # Use proper analysis engine
            return self.engine.evaluate_move(game, move, perspective)
    
    def get_best_move_and_alternatives(self, game: HexGame, top_k=5, use_deep=False):
        """Get best move and top alternatives with ratings"""
        if isinstance(self.agent, NeuralMCTSAgent):
            # For Neural MCTS, use the agent's move selection
            perspective = game.turn
            valid_moves = game.valid_moves()
            if not valid_moves:
                return None, []
            
            move_scores = []
            for mv in valid_moves:
                value, rating_str = self.evaluate_move(game, mv, perspective)
                move_scores.append((mv, value, rating_str))
            
            move_scores.sort(key=lambda x: x[1], reverse=True)
            best_move = move_scores[0][0] if move_scores else None
            alternatives = move_scores[:top_k]
            return best_move, alternatives
        else:
            # Use faster heuristic-based evaluation for UI responsiveness
            perspective = game.turn
            valid_moves = game.valid_moves()
            if not valid_moves:
                return None, []
            
            # Quick checks first
            # Check for immediate win
            for mv in valid_moves:
                test_game = game.clone()
                test_game.make_move(*mv)
                if test_game.terminal and test_game.winner == perspective:
                    return mv, [(mv, 1e6, "Win!")]
            
            # Check for immediate block
            opponent = HexGame.P1 if perspective == HexGame.P2 else HexGame.P2
            opponent_threats = []
            for mv in valid_moves:
                test_game = game.clone()
                test_game.turn = opponent
                if test_game.is_valid_move(*mv):
                    test_game.make_move(*mv)
                    if test_game.terminal and test_game.winner == opponent:
                        opponent_threats.append(mv)
            
            if opponent_threats:
                return opponent_threats[0], [(opponent_threats[0], 500, "Block")]
            
            # Ultra-fast evaluation for UI - only evaluate top 8 moves
            move_scores = []
            # Quick pre-sort by simple distance heuristic
            scored_pre = []
            center = (game.size - 1) / 2.0
            for mv in valid_moves[:min(20, len(valid_moves))]:  # Limit initial candidates
                # Simple heuristic: path distance + center
                test_game = game.clone()
                test_game.make_move(*mv)
                my_dist = shortest_path_distance_cached(test_game, perspective)
                opp = HexGame.P1 if perspective == HexGame.P2 else HexGame.P2
                opp_dist = shortest_path_distance_cached(test_game, opp)
                
                if my_dist < 10**7 and opp_dist < 10**7:
                    path_diff = float(opp_dist - my_dist)
                else:
                    path_diff = 0.0
                
                center_bonus = -0.1 * (abs(mv[0] - center) + abs(mv[1] - center))
                quick_score = path_diff + center_bonus
                scored_pre.append((quick_score, mv))
            
            scored_pre.sort(reverse=True, key=lambda x: x[0])
            candidate_moves = [mv for _, mv in scored_pre[:8]]  # Only top 8
            
            for mv in candidate_moves:
                test_game = game.clone()
                test_game.make_move(*mv)
                
                # Ultra-fast evaluation - only shortest path
                value = self._fast_expert_eval(test_game, perspective)
                rating = value * 1000
                
                if abs(rating) < 10:
                    rating_str = f"{rating:+.1f}"
                else:
                    rating_str = f"{rating:+.0f}"
                
                move_scores.append((mv, value, rating_str))
            
            move_scores.sort(key=lambda x: x[1], reverse=True)
            best_move = move_scores[0][0] if move_scores else None
            alternatives = move_scores[:top_k]
            
            return best_move, alternatives
    
    def _fast_expert_eval(self, game: HexGame, perspective):
        """Ultra-fast evaluation for real-time UI - only uses cached shortest path"""
        if game.terminal:
            return 1.0 if game.winner == perspective else -1.0
        
        opp = HexGame.P1 if perspective == HexGame.P2 else HexGame.P2
        
        # ONLY use cached shortest path - this is instant
        myd = shortest_path_distance_cached(game, perspective)
        oppd = shortest_path_distance_cached(game, opp)
        
        if myd >= 10**7 and oppd >= 10**7:
            return 0.0
        
        # Simple normalized score based on path difference
        score = float(oppd - myd)
        # Normalize to [-1, 1] range (assuming max distance is around 20 for 11x11)
        return max(-1.0, min(1.0, score / 20.0))
    
    def _get_fast_candidates(self, game: HexGame, valid_moves, k=12):
        """Get top candidate moves using fast heuristics"""
        if len(valid_moves) <= k:
            return valid_moves
        
        scored = []
        current_player = game.turn
        center = (game.size - 1) / 2.0
        
        for r, c in valid_moves:
            # Fast path distance check (uses cache)
            test_game = game.clone()
            test_game.make_move(r, c)
            my_dist = shortest_path_distance_cached(test_game, current_player)
            opp = HexGame.P1 if current_player == HexGame.P2 else HexGame.P2
            opp_dist = shortest_path_distance_cached(test_game, opp)
            
            if my_dist < 10**7 and opp_dist < 10**7:
                path_score = float(opp_dist - my_dist)
            else:
                path_score = 0.0
            
            # Center preference
            center_score = -0.1 * (abs(r - center) + abs(c - center))
            
            # Connectivity (count own stones nearby)
            connectivity = 0.0
            for nr, nc in game.neighbors(r, c):
                if game.board[nr][nc] == current_player:
                    connectivity += 1.0
            
            total_score = path_score + center_score + connectivity
            scored.append((total_score, (r, c)))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        return [mv for _, mv in scored[:k]]

# -------------------- Enhanced GUI --------------------
def hexagon_points(cx,cy,r):
    pts=[]
    for i in range(6):
        ang=math.pi/180*(60*i-30)
        pts.append((cx+r*math.cos(ang), cy+r*math.sin(ang)))
    return pts

def render_game(screen, game, size, CELL, MARGIN):
    """Simple render function for training visualization"""
    if not screen: return
    screen.fill((240,240,240))
    for r in range(size):
        for c in range(size):
            x=MARGIN + c*CELL + r*(CELL//2)
            y=MARGIN + r*int(CELL*0.86)
            pygame.draw.polygon(screen, (200,200,200), hexagon_points(x,y,CELL//2), 2)
            v=game.board[r][c]
            if v==HexGame.P1:
                pygame.draw.polygon(screen, (200,50,50), hexagon_points(x,y,CELL//2))
            elif v==HexGame.P2:
                pygame.draw.polygon(screen, (50,50,200), hexagon_points(x,y,CELL//2))
    pygame.display.flip()

def render_game_enhanced(screen, game, size, CELL, margin_x, margin_y, panel_width, 
                         evaluator=None, suggested_move=None, 
                         move_history_ratings=None, current_eval=None, hover_pos=None):
    """Optimized rendering with caching and frame skipping"""
    global GUI_UPDATE_CACHE, GUI_LAST_RENDER_TIME
    
    if not screen: 
        return margin_x*2 + CELL*size + CELL//2*(size-1)
    
    # Frame skipping for performance
    current_time = time.time()
    if current_time - GUI_LAST_RENDER_TIME < 1.0 / (GUI_FPS / GUI_FRAME_SKIP):
        return 0
    GUI_LAST_RENDER_TIME = current_time
    
    # Create cache key for board state
    board_hash = game.board_hash()
    cache_key = (board_hash, suggested_move, hover_pos)
    
    # Check if we can reuse cached rendering
    if cache_key in GUI_UPDATE_CACHE and not hover_pos:
        cached_surface = GUI_UPDATE_CACHE[cache_key]
        screen.blit(cached_surface, (0, 0))
        return 0
    
    # Colors - Premium Hex Game styling
    BG_COLOR = (18, 18, 22)  # Deep dark background
    BOARD_BG = (32, 32, 38)  # Rich board background
    HEX_BORDER = (80, 80, 90)  # Subtle borders
    HEX_EMPTY = (50, 50, 58)  # Empty hex color
    HEX_HOVER = (70, 70, 82)  # Hover effect
    P1_COLOR = (235, 90, 90)  # Vibrant Red
    P1_SHADOW = (180, 50, 50)  # Shadow for depth
    P2_COLOR = (90, 140, 235)  # Vibrant Blue
    P2_SHADOW = (50, 90, 180)  # Shadow for depth
    HIGHLIGHT_COLOR = (255, 220, 50)  # Bright gold for suggestions
    HIGHLIGHT_GLOW = (255, 240, 100)  # Glow effect
    HIGHLIGHT_BORDER = (255, 200, 0)
    TEXT_COLOR = (250, 250, 250)  # Bright text
    PANEL_COLOR = (28, 28, 34)  # Panel background
    PANEL_BORDER = (45, 45, 52)  # Panel border
    ACCENT_COLOR = (100, 150, 255)  # Accent color
    
    # Create a surface for the board to enable caching
    board_surface = pygame.Surface((screen.get_width(), screen.get_height()))
    board_surface.fill(BG_COLOR)
    
    # Draw board background with gradient effect
    board_area = pygame.Rect(margin_x, margin_y, 
                             CELL*size + CELL//2*(size-1), 
                             int(CELL*0.86)*(size-1) + CELL)
    pygame.draw.rect(board_surface, BOARD_BG, board_area)
    # Add subtle border
    pygame.draw.rect(board_surface, PANEL_BORDER, board_area, 3)
    
    # Pre-calculate hexagon points for performance
    hex_points_cache = {}
    for r in range(size):
        for c in range(size):
            x = margin_x + c*CELL + r*(CELL//2)
            y = margin_y + r*int(CELL*0.86)
            hex_points_cache[(r, c)] = hexagon_points(x, y, CELL//2)
    
    # Draw hexagons with optimized rendering
    for r in range(size):
        for c in range(size):
            hex_pts = hex_points_cache[(r, c)]
            v = game.board[r][c]
            is_suggested = suggested_move and (r, c) == suggested_move and v == HexGame.EMPTY
            is_hovered = hover_pos and (r, c) == hover_pos and v == HexGame.EMPTY
            
            # Draw base hexagon first
            if v == HexGame.P1:
                # Red piece with shadow effect
                shadow_offset = 2
                shadow_pts = [(px + shadow_offset, py + shadow_offset) for px, py in hex_pts]
                pygame.draw.polygon(board_surface, P1_SHADOW, shadow_pts)
                pygame.draw.polygon(board_surface, P1_COLOR, hex_pts)
                pygame.draw.polygon(board_surface, (255, 120, 120), hex_pts, 2)
            elif v == HexGame.P2:
                # Blue piece with shadow effect
                shadow_offset = 2
                shadow_pts = [(px + shadow_offset, py + shadow_offset) for px, py in hex_pts]
                pygame.draw.polygon(board_surface, P2_SHADOW, shadow_pts)
                pygame.draw.polygon(board_surface, P2_COLOR, hex_pts)
                pygame.draw.polygon(board_surface, (120, 160, 255), hex_pts, 2)
            else:
                # Empty hex with subtle styling
                pygame.draw.polygon(board_surface, HEX_EMPTY, hex_pts)
                pygame.draw.polygon(board_surface, HEX_BORDER, hex_pts, 1)
            
            # Draw highlights on top - only for empty cells to avoid overlap
            if is_suggested and v == HexGame.EMPTY:
                # Calculate center of hexagon for glow
                cx = sum(pt[0] for pt in hex_pts) / len(hex_pts)
                cy = sum(pt[1] for pt in hex_pts) / len(hex_pts)
                # Outer glow (only for empty cells)
                glow_pts = hexagon_points(cx, cy, CELL//2 + 4)
                pygame.draw.polygon(board_surface, HIGHLIGHT_GLOW, glow_pts, 0)
                # Main highlight with thicker border (draw on top of empty hex)
                pygame.draw.polygon(board_surface, HIGHLIGHT_COLOR, hex_pts)
                pygame.draw.polygon(board_surface, HIGHLIGHT_BORDER, hex_pts, 4)
            elif is_hovered and v == HexGame.EMPTY:
                # Hover effect (only for empty cells)
                pygame.draw.polygon(board_surface, HEX_HOVER, hex_pts)
                pygame.draw.polygon(board_surface, ACCENT_COLOR, hex_pts, 2)
    
    # Blit the board surface to screen
    screen.blit(board_surface, (0, 0))
    
    # Cache the rendered surface if no hover
    if not hover_pos and len(GUI_UPDATE_CACHE) < 10:  # Limit cache size
        GUI_UPDATE_CACHE[cache_key] = board_surface.copy()
    
    return 0

def show_bot_selection_menu(screen, screen_width, screen_height, net=None):
    """Display bot selection menu and return selected bot"""
    if not PYGAME_AVAILABLE:
        return None
    
    BG_COLOR = (18, 18, 22)
    TEXT_COLOR = (250, 250, 250)
    ACCENT_COLOR = (100, 150, 255)
    BUTTON_COLOR = (40, 40, 48)
    BUTTON_HOVER = (60, 60, 70)
    BUTTON_BORDER = (80, 80, 90)
    BEGINNER_COLOR = (90, 140, 235)  # Blue
    INTERMEDIATE_COLOR = (255, 200, 50)  # Yellow/Gold
    EXPERT_COLOR = (235, 90, 90)  # Red
    
    font_title = pygame.font.SysFont("Arial", 48, bold=True)
    font_large = pygame.font.SysFont("Arial", 32, bold=True)
    font_medium = pygame.font.SysFont("Arial", 24)
    font_small = pygame.font.SysFont("Arial", 18)
    
    buttons = []
    button_height = 80
    button_width = 500
    button_spacing = 20
    start_y = screen_height // 2 - 150
    
    # Beginner Bot button
    buttons.append({
        'rect': pygame.Rect(screen_width // 2 - button_width // 2, start_y, button_width, button_height),
        'text': '🔵 Beginner Bot - Pattern Follower',
        'desc': 'Local play, no global strategy',
        'color': BEGINNER_COLOR,
        'bot_type': 'beginner'
    })
    
    # Intermediate Bot button
    buttons.append({
        'rect': pygame.Rect(screen_width // 2 - button_width // 2, start_y + button_height + button_spacing, button_width, button_height),
        'text': '🟡 Intermediate Bot - Shape Builder',
        'desc': 'Path formation, bridge recognition, threat blocking',
        'color': INTERMEDIATE_COLOR,
        'bot_type': 'intermediate'
    })
    
    # Expert Bot button
    buttons.append({
        'rect': pygame.Rect(screen_width // 2 - button_width // 2, start_y + 2 * (button_height + button_spacing), button_width, button_height),
        'text': '🔴 Expert Bot - Grandmaster',
        'desc': 'Neural MCTS, virtual connections, expert-level play',
        'color': EXPERT_COLOR,
        'bot_type': 'expert'
    })
    
    selected_bot = None
    clock = pygame.time.Clock()
    
    while selected_bot is None:
        screen.fill(BG_COLOR)
        
        # Title
        title = font_title.render("Select Your Opponent", True, ACCENT_COLOR)
        title_rect = title.get_rect(center=(screen_width // 2, 150))
        screen.blit(title, title_rect)
        
        # Get mouse position
        mouse_pos = pygame.mouse.get_pos()
        
        # Draw buttons
        for button in buttons:
            is_hovered = button['rect'].collidepoint(mouse_pos)
            color = BUTTON_HOVER if is_hovered else BUTTON_COLOR
            
            # Draw button background
            pygame.draw.rect(screen, color, button['rect'])
            pygame.draw.rect(screen, button['color'], button['rect'], 3)
            
            # Draw button text
            text_surface = font_large.render(button['text'], True, TEXT_COLOR)
            text_rect = text_surface.get_rect(center=(button['rect'].centerx, button['rect'].centery - 10))
            screen.blit(text_surface, text_rect)
            
            # Draw description
            desc_surface = font_small.render(button['desc'], True, (180, 180, 180))
            desc_rect = desc_surface.get_rect(center=(button['rect'].centerx, button['rect'].centery + 20))
            screen.blit(desc_surface, desc_rect)
        
        # Instructions
        inst_text = font_medium.render("Click a bot to start playing", True, (150, 150, 150))
        inst_rect = inst_text.get_rect(center=(screen_width // 2, screen_height - 100))
        screen.blit(inst_text, inst_rect)
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    for button in buttons:
                        if button['rect'].collidepoint(event.pos):
                            selected_bot = button['bot_type']
                            break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                # Number keys for quick selection
                if event.key == pygame.K_1:
                    selected_bot = 'beginner'
                elif event.key == pygame.K_2:
                    selected_bot = 'intermediate'
                elif event.key == pygame.K_3:
                    selected_bot = 'expert'
        
        clock.tick(60)
    
    # Create the selected bot
    if selected_bot == 'beginner':
        return BeginnerBot()
    elif selected_bot == 'intermediate':
        return IntermediateBot()
    elif selected_bot == 'expert':
        if net is None:
            print("[ERROR] Expert bot requires neural network. Falling back to Intermediate.")
            return IntermediateBot()
        return ExpertBot(net, simulations=MCTS_SIMS)
    
    return None

def pygame_gui(agent=None, size=BOARD_SIZE, net=None):
    if not PYGAME_AVAILABLE:
        print("[GUI] Pygame not available; falling back to CLI."); return cli_play(True,size,agent)
    pygame.init()
    
    # Get screen dimensions for fullscreen
    info = pygame.display.Info()
    screen_width = info.current_w
    screen_height = info.current_h
    
    # Calculate optimal cell size and margins for fullscreen
    PANEL_WIDTH = int(screen_width * 0.25)  # 25% for panel
    available_width = screen_width - PANEL_WIDTH
    available_height = screen_height
    
    # Calculate cell size to fit board nicely
    max_cell_by_width = (available_width - 100) / (size + (size-1)*0.5)
    max_cell_by_height = (available_height - 100) / (size * 0.86)
    CELL = int(min(max_cell_by_width, max_cell_by_height, 60))  # Max 60px
    CELL = max(CELL, 35)  # Min 35px
    
    MARGIN = 50
    board_width = MARGIN*2 + CELL*size + CELL//2*(size-1)
    board_height = MARGIN*2 + int(CELL*0.86)*(size-1) + CELL
    
    # Center the board
    board_offset_x = (available_width - board_width) // 2
    board_offset_y = (available_height - board_height) // 2
    
    os.environ['SDL_VIDEO_CENTERED']='1'
    try:
        # Fullscreen mode
        screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
        pygame.display.set_caption("Hex Game Engine")
    except Exception as e:
        print(f"[GUI] Failed to init fullscreen: {e}, trying windowed mode")
        try:
            screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("Hex Game Engine")
        except Exception as e2:
            print(f"[GUI] Failed to init display: {e2}"); return cli_play(True,size,agent)
    
    # Show bot selection menu if no agent provided
    if agent is None:
        agent = show_bot_selection_menu(screen, screen_width, screen_height, net)
        if agent is None:
            pygame.quit()
            return
    
    # Fonts - larger for fullscreen
    font_title = pygame.font.SysFont("Arial", 36, bold=True)
    font_large = pygame.font.SysFont("Arial", 32, bold=True)
    font_medium = pygame.font.SysFont("Arial", 24)
    font_small = pygame.font.SysFont("Arial", 18)
    font_tiny = pygame.font.SysFont("Arial", 16)
    
    clock=pygame.time.Clock()
    game=HexGame(size=size, swap_rule=True); human=HexGame.P1; bot=agent
    
    # Move evaluation system
    evaluator = None
    if agent and hasattr(agent, 'select_move'):
        # Only create evaluator for AI agents that have select_move method
        evaluator = MoveEvaluator(agent)
    suggested_move = None
    move_history_ratings = []  # List of (move_num, move, rating_str, player)
    current_eval = None
    show_suggestions = True
    panel_scroll_offset = 0  # Scroll offset for entire panel
    
    # Colors
    TEXT_COLOR = (250, 250, 250)
    HIGHLIGHT_COLOR = (255, 220, 50)
    GOOD_MOVE_COLOR = (120, 220, 120)
    BAD_MOVE_COLOR = (220, 120, 120)
    ACCENT_COLOR = (100, 150, 255)
    hover_pos = None  # Track hover position
    
    # Performance optimization
    last_ai_think_time = 0
    ai_thinking = False
    ai_move = None
    frame_count = 0
    eval_update_interval = 30  # Update evaluation every 30 frames for better performance
    last_eval_frame = 0
    panel_surface = None  # Cache for panel to prevent flickering
    panel_dirty = True  # Flag to indicate panel needs redraw
    
    def update_evaluation():
        """Update evaluation in background for better performance"""
        nonlocal current_eval, suggested_move, last_eval_frame
        # Only update evaluation periodically to improve performance
        if frame_count - last_eval_frame < eval_update_interval:
            return
            
        if evaluator and not game.terminal and game.turn == human:
            try:
                # Use much lighter evaluation for better performance
                current_eval = evaluator.evaluate_position(game, human, use_deep=False)
                if show_suggestions:
                    # Get best move with lighter evaluation
                    suggested_move = evaluator.get_best_move(game)
                last_eval_frame = frame_count
            except Exception as e:
                current_eval = None
                suggested_move = None
    
    def draw_panel(panel_x, y_offset=30):
        """Draw the side panel with analysis - FULLY SCROLLABLE"""
        nonlocal panel_scroll_offset, panel_surface, panel_dirty
        
        # Create a virtual surface that's much taller (can scroll)
        VIRTUAL_PANEL_HEIGHT = 5000  # Large virtual height for scrolling
        
        # Create or reuse panel surface (virtual surface)
        if panel_surface is None or panel_surface.get_size() != (PANEL_WIDTH, VIRTUAL_PANEL_HEIGHT):
            panel_surface = pygame.Surface((PANEL_WIDTH, VIRTUAL_PANEL_HEIGHT))
            panel_dirty = True
        
        # Always redraw when dirty or scroll changed
        if not hasattr(draw_panel, '_last_scroll'):
            draw_panel._last_scroll = -1
        
        scroll_changed = (panel_scroll_offset != draw_panel._last_scroll)
        if not panel_dirty and not scroll_changed:
            # Just blit the visible portion
            visible_rect = pygame.Rect(0, panel_scroll_offset, PANEL_WIDTH, min(screen_height, panel_surface.get_height() - panel_scroll_offset))
            if visible_rect.height > 0 and visible_rect.y >= 0 and visible_rect.y + visible_rect.height <= panel_surface.get_height():
                try:
                    visible_surface = panel_surface.subsurface(visible_rect)
                    screen.blit(visible_surface, (panel_x, 0))
                except:
                    screen.blit(panel_surface, (panel_x, -panel_scroll_offset))
            return
        
        draw_panel._last_scroll = panel_scroll_offset
        
        # Clear virtual panel surface
        panel_surface.fill((18, 18, 22))
        
        y = y_offset
        
        # Title with accent
        title = font_title.render("Hex Game Engine", True, ACCENT_COLOR)
        panel_surface.blit(title, (25, y))
        y += 60
        
        # Bot name display
        if bot and hasattr(bot, 'name'):
            bot_name_text = font_medium.render(f"vs {bot.name}", True, ACCENT_COLOR)
            panel_surface.blit(bot_name_text, (25, y))
            y += 40
        
        # Subtitle
        subtitle = font_small.render("Analysis & Evaluation", True, (180, 180, 180))
        panel_surface.blit(subtitle, (25, y))
        y += 50
        
        # Controls info - moved up and fixed positioning
        controls_label = font_medium.render("Controls", True, TEXT_COLOR)
        panel_surface.blit(controls_label, (25, y))
        y += 35
        
        # Control hints with proper spacing
        hint1 = font_small.render("[S] Toggle Suggestions", True, (180, 180, 180))
        panel_surface.blit(hint1, (25, y))
        y += 30
        
        hint2 = font_small.render("[A] Analyze Position", True, (180, 180, 180))
        panel_surface.blit(hint2, (25, y))
        y += 30
        
        hint3 = font_small.render("[↑↓] or [Wheel] Scroll", True, (180, 180, 180))
        panel_surface.blit(hint3, (25, y))
        y += 30
        
        hint4 = font_small.render("[F12] Screenshot", True, (180, 180, 180))
        panel_surface.blit(hint4, (25, y))
        y += 50
        
        # Separator line
        pygame.draw.line(panel_surface, (60, 60, 70), (25, y), (295, y), 1)
        y += 30
        
        # Current evaluation with styled box
        if current_eval is not None:
            eval_label = font_medium.render("Position Evaluation", True, TEXT_COLOR)
            panel_surface.blit(eval_label, (25, y))
            y += 40
            
            eval_value = current_eval * 1000
            if abs(eval_value) < 10:
                eval_str = f"{eval_value:+.1f}"
            else:
                eval_str = f"{eval_value:+.0f}"
            
            eval_color = GOOD_MOVE_COLOR if eval_value > 0 else BAD_MOVE_COLOR if eval_value < 0 else TEXT_COLOR
            
            # Draw evaluation box
            eval_box = pygame.Rect(25, y, 280, 50)
            pygame.draw.rect(panel_surface, (40, 40, 48), eval_box)
            pygame.draw.rect(panel_surface, eval_color, eval_box, 2)
            
            eval_text = font_large.render(eval_str, True, eval_color)
            text_rect = eval_text.get_rect(center=(145, y + 25))
            panel_surface.blit(eval_text, text_rect)
            y += 70
            
            # Fluid position bar showing which side is winning
            bar_label = font_medium.render("Position Balance", True, TEXT_COLOR)
            panel_surface.blit(bar_label, (25, y))
            y += 40
            
            # Bar dimensions
            bar_width = 280
            bar_height = 30
            bar_x = 25
            bar_y = y
            
            # Background bar (neutral gray)
            bar_bg = pygame.Rect(bar_x, bar_y, bar_width, bar_height)
            pygame.draw.rect(panel_surface, (50, 50, 58), bar_bg)
            pygame.draw.rect(panel_surface, (80, 80, 90), bar_bg, 2)
            
            # Calculate bar fill based on evaluation (clamp to [-1, 1])
            eval_normalized = max(-1.0, min(1.0, current_eval))
            center_x = bar_x + bar_width // 2
            
            # Draw center line
            pygame.draw.line(panel_surface, (100, 100, 110), (center_x, bar_y), (center_x, bar_y + bar_height), 2)
            
            # Draw fluid bar based on evaluation
            if eval_normalized > 0:
                # P1 (Human) is winning - bar extends right from center
                fill_width = int((bar_width // 2) * abs(eval_normalized))
                fill_rect = pygame.Rect(center_x, bar_y, fill_width, bar_height)
                # Gradient effect for P1 (red)
                for i in range(fill_width):
                    alpha = 1.0 - (i / fill_width) * 0.3
                    color = (int(235 * alpha), int(90 * alpha), int(90 * alpha))
                    line_rect = pygame.Rect(center_x + i, bar_y, 1, bar_height)
                    pygame.draw.rect(panel_surface, color, line_rect)
                # Border
                pygame.draw.rect(panel_surface, (255, 120, 120), fill_rect, 2)
            elif eval_normalized < 0:
                # P2 (Bot) is winning - bar extends left from center
                fill_width = int((bar_width // 2) * abs(eval_normalized))
                fill_rect = pygame.Rect(center_x - fill_width, bar_y, fill_width, bar_height)
                # Gradient effect for P2 (blue)
                for i in range(fill_width):
                    alpha = 1.0 - (i / fill_width) * 0.3
                    color = (int(90 * alpha), int(140 * alpha), int(235 * alpha))
                    line_rect = pygame.Rect(center_x - i, bar_y, 1, bar_height)
                    pygame.draw.rect(panel_surface, color, line_rect)
                # Border
                pygame.draw.rect(panel_surface, (120, 160, 255), fill_rect, 2)
            
            # Labels on sides
            p1_label = font_small.render("You", True, (235, 90, 90))
            p2_label = font_small.render("Bot", True, (90, 140, 235))
            panel_surface.blit(p1_label, (bar_x + bar_width - 40, bar_y + bar_height + 5))
            panel_surface.blit(p2_label, (bar_x, bar_y + bar_height + 5))
            
            y += 60
        
        # Best move suggestion - only show if suggestions are ON
        if show_suggestions and suggested_move and not game.terminal:
            suggestion_label = font_medium.render("Best Move", True, TEXT_COLOR)
            panel_surface.blit(suggestion_label, (25, y))
            y += 40
            
            mv_str = f"({suggested_move[0]},{suggested_move[1]})"
            sugg_box = pygame.Rect(25, y, 280, 50)
            pygame.draw.rect(panel_surface, (50, 45, 35), sugg_box)
            pygame.draw.rect(panel_surface, HIGHLIGHT_COLOR, sugg_box, 3)
            
            mv_text = font_large.render(mv_str, True, HIGHLIGHT_COLOR)
            text_rect = mv_text.get_rect(center=(145, y + 25))
            panel_surface.blit(mv_text, text_rect)
            y += 70
        
        # Separator
        pygame.draw.line(panel_surface, (60, 60, 70), (25, y), (295, y), 1)
        y += 30
        
        # Move History section
        history_label = font_medium.render("Move History", True, ACCENT_COLOR)
        panel_surface.blit(history_label, (25, y))
        y += 50
        
        # Draw ALL moves - simple format, better spacing
        if move_history_ratings:
            for move_num, move, rating, player in move_history_ratings:
                # Simple format: just (r,c)
                move_str = f"({move[0]},{move[1]})"
                move_text = font_large.render(move_str, True, (255, 255, 255))
                text_x = 25 + (280 - move_text.get_width()) // 2
                panel_surface.blit(move_text, (text_x, y))
                y += 45  # More spacing for readability
        else:
            no_history_text = font_small.render("Press [A] to analyze", True, (150, 150, 150))
            panel_surface.blit(no_history_text, (25, y))
            y += 40
        
        # Instructions at the end of virtual panel
        y += 30
        inst_bg = pygame.Rect(25, y - 10, 280, 100)
        pygame.draw.rect(panel_surface, (35, 35, 42), inst_bg)
        pygame.draw.rect(panel_surface, ACCENT_COLOR, inst_bg, 2)
        
        inst1 = font_small.render("• Click hex to play", True, (220, 220, 220))
        inst2 = font_small.render("• ESC: Exit game", True, (220, 220, 220))
        inst3 = font_small.render("• F12: Screenshot", True, (220, 220, 220))
        panel_surface.blit(inst1, (30, y))
        panel_surface.blit(inst2, (30, y + 30))
        panel_surface.blit(inst3, (30, y + 60))
        
        # Calculate total content height
        total_content_height = y + 100
        
        # Clamp scroll offset to valid range
        max_scroll = max(0, total_content_height - screen_height)
        if panel_scroll_offset > max_scroll:
            panel_scroll_offset = max_scroll
        if panel_scroll_offset < 0:
            panel_scroll_offset = 0
        
        # Blit only the visible portion of the virtual panel
        visible_rect = pygame.Rect(0, panel_scroll_offset, PANEL_WIDTH, min(screen_height, panel_surface.get_height() - panel_scroll_offset))
        if visible_rect.height > 0 and visible_rect.y >= 0 and visible_rect.y + visible_rect.height <= panel_surface.get_height():
            try:
                visible_surface = panel_surface.subsurface(visible_rect)
                screen.blit(visible_surface, (panel_x, 0))
            except:
                # Fallback: just blit the whole surface if subsurface fails
                screen.blit(panel_surface, (panel_x, -panel_scroll_offset))
        
        panel_dirty = False
    
    # Cache for evaluation to avoid recomputing - using MoveEvaluator's internal cache now
    
    def sync_move_history():
        """Sync move_history_ratings with game.move_history - show all moves"""
        nonlocal panel_dirty
        # Create a map of move numbers to existing entries
        existing_by_num = {num: (num, move, rating, player) for num, move, rating, player in move_history_ratings}
        
        # Update or add moves from game.move_history
        for move_num, (player, move) in enumerate(game.move_history, 1):
            if move_num in existing_by_num:
                # Move already exists, keep it (might have rating from analysis)
                continue
            else:
                # New move, add it without rating
                move_history_ratings.append((move_num, move, "", player))
        
        # Sort by move number to keep order
        move_history_ratings.sort(key=lambda x: x[0])
        panel_dirty = True
    
    def analyze_move(move, player):
        """Analyze a move and add to history - optimized version"""
        if not evaluator:
            return
        
        # Ensure move is a tuple
        if not isinstance(move, (tuple, list)) or len(move) != 2:
            return
        
        # Simplified analysis - just evaluate the current position
        try:
            value, rating_str = evaluator.evaluate_move(game, move, player)
            # Find if move already exists in history
            move_num = len(game.move_history)
            # Update or add the move
            for idx, (num, mv, _, pl) in enumerate(move_history_ratings):
                if mv == move and num == move_num:
                    move_history_ratings[idx] = (num, move, rating_str, player)
                    return
            # If not found, add it
            move_history_ratings.append((move_num, move, rating_str, player))
        except:
            pass  # Skip analysis if it fails
    
    def analyze_all_moves():
        """Analyze all moves in the game history"""
        nonlocal panel_scroll_offset, panel_dirty, current_eval
        
        if not evaluator:
            print("[ANALYZE] No evaluator available")
            return
        
        if not game.move_history:
            print("[ANALYZE] No moves to analyze")
            return
        
        print(f"[ANALYZE] Analyzing {len(game.move_history)} moves...")
        
        # Clear existing history
        move_history_ratings.clear()
        
        # Replay the game and analyze each move
        temp_game = HexGame(size=size, swap_rule=True)
        move_num = 1
        max_moves = len(game.move_history)
        
        for idx, (player, move) in enumerate(game.move_history):
            # Safety check to prevent infinite loops
            if idx >= max_moves or move_num > max_moves * 2:
                print(f"[ANALYZE] Safety limit reached at move {move_num}")
                break
                
            # Analyze the move in the context of the game state before it was made
            try:
                # Clone the temp game to analyze the move
                test_game = temp_game.clone()
                value, rating_str = evaluator.evaluate_move(test_game, move, player)
                move_history_ratings.append((move_num, move, rating_str, player))
                move_num += 1
            except Exception as e:
                # If analysis fails, just record the move without rating
                print(f"[ANALYZE] Error analyzing move {move_num}: {e}")
                move_history_ratings.append((move_num, move, "N/A", player))
                move_num += 1
            
            # Make the move in temp game to advance state
            try:
                temp_game.make_move(*move)
            except Exception as e:
                print(f"[ANALYZE] Error making move in replay: {e}")
                break  # Stop if we can't make the move
        
        # Update current evaluation
        try:
            current_eval = evaluator.evaluate_position(game, human, use_deep=False)
        except:
            pass
        
        # Reset scroll to top when analyzing
        panel_scroll_offset = 0
        panel_dirty = True
        print(f"[ANALYZE] Analysis complete: {len(move_history_ratings)} moves analyzed")
    
    def draw():
        # Render board with centered margins
        render_game_enhanced(screen, game, size, CELL, board_offset_x, board_offset_y, PANEL_WIDTH,
                                      evaluator, suggested_move, move_history_ratings, current_eval, hover_pos)
        # Draw panel on the right
        panel_x = screen_width - PANEL_WIDTH
        # Draw panel border
        pygame.draw.line(screen, (45, 45, 52), (panel_x, 0), (panel_x, screen.get_height()), 2)
        draw_panel(panel_x)
        pygame.display.flip()
    
    # Initial evaluation
    update_evaluation()
    # Sync initial move history if game already has moves
    sync_move_history()
    running=True
    max_frames = 1000000  # Safety limit to prevent infinite loops
    frame_safety_count = 0
    
    while running:
        frame_count += 1
        frame_safety_count += 1
        
        # Safety check to prevent infinite loops
        if frame_safety_count > max_frames:
            print("[GUI] Safety limit reached, exiting game loop")
            running = False
            break
        
        # Optimized rendering - redraw when needed
        need_redraw = (frame_count % GUI_FRAME_SKIP == 0) or ai_thinking or hover_pos is not None or panel_dirty
        
        if need_redraw:
            draw()
        
        # Process events efficiently
        for event in pygame.event.get():
            if event.type==pygame.QUIT: 
                running=False; break
            
            if event.type==pygame.KEYDOWN:
                if event.key == pygame.K_s:  # Toggle suggestions
                    show_suggestions = not show_suggestions
                    update_evaluation()
                    panel_dirty = True
                    need_redraw = True
                elif event.key == pygame.K_a:  # Analyze position
                    # Analyze all moves in history
                    analyze_all_moves()
                    update_evaluation()
                    panel_dirty = True
                    need_redraw = True
                    draw()  # Force immediate redraw
                elif event.key == pygame.K_UP:  # Scroll panel up
                    old_offset = panel_scroll_offset
                    panel_scroll_offset = max(0, panel_scroll_offset - 50)
                    if old_offset != panel_scroll_offset:
                        panel_dirty = True
                        need_redraw = True
                        draw()
                elif event.key == pygame.K_DOWN:  # Scroll panel down
                    old_offset = panel_scroll_offset
                    # Calculate max scroll based on content
                    estimated_content_height = 300 + len(move_history_ratings) * 45
                    max_scroll = max(0, estimated_content_height - screen_height)
                    panel_scroll_offset = min(max_scroll, panel_scroll_offset + 50)
                    if old_offset != panel_scroll_offset:
                        panel_dirty = True
                        need_redraw = True
                        draw()
                elif event.key == pygame.K_F12:  # Screenshot
                    try:
                        # Create screenshot directory if it doesn't exist
                        screenshot_dir = "screenshots"
                        if not os.path.exists(screenshot_dir):
                            os.makedirs(screenshot_dir)
                        
                        # Generate filename with timestamp
                        from datetime import datetime
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = os.path.join(screenshot_dir, f"hex_game_{timestamp}.png")
                        
                        # Save screenshot
                        pygame.image.save(screen, filename)
                        print(f"[SCREENSHOT] Saved to: {filename}")
                        
                        # Show brief confirmation message
                        confirm_font = pygame.font.SysFont("Arial", 24, bold=True)
                        confirm_text = confirm_font.render("Screenshot saved!", True, (0, 255, 0))
                        confirm_rect = confirm_text.get_rect(center=(screen_width // 2, 50))
                        screen.blit(confirm_text, confirm_rect)
                        pygame.display.flip()
                        time.sleep(1)  # Show message for 1 second
                        panel_dirty = True
                        need_redraw = True
                    except Exception as e:
                        print(f"[SCREENSHOT] Error saving screenshot: {e}")
                elif event.key == pygame.K_ESCAPE:  # Exit fullscreen
                    running = False
                    break
            
            # Mouse wheel scrolling for entire panel
            if event.type == pygame.MOUSEWHEEL:
                # Check if mouse is over panel
                mx, my = event.pos if hasattr(event, 'pos') else pygame.mouse.get_pos()
                if mx >= screen_width - PANEL_WIDTH:
                    scroll_amount = event.y
                    old_offset = panel_scroll_offset
                    if scroll_amount > 0:  # Scroll up
                        panel_scroll_offset = max(0, panel_scroll_offset - 50)
                    elif scroll_amount < 0:  # Scroll down
                        estimated_content_height = 300 + len(move_history_ratings) * 45
                        max_scroll = max(0, estimated_content_height - screen_height)
                        panel_scroll_offset = min(max_scroll, panel_scroll_offset + 50)
                    if old_offset != panel_scroll_offset:
                        panel_dirty = True
                        need_redraw = True
                        draw()
            
            # Optimized mouse hover tracking
            if event.type == pygame.MOUSEMOTION:
                mx, my = event.pos
                old_hover = hover_pos
                hover_pos = None
                # Check if mouse is over board
                if mx < screen_width - PANEL_WIDTH:
                    best = None
                    bestd = 1e9
                    # Use cached positions for better performance
                    for r in range(size):
                        for c in range(size):
                            x = board_offset_x + c*CELL + r*(CELL//2)
                            y = board_offset_y + r*int(CELL*0.86)
                            d = (mx-x)**2 + (my-y)**2
                            if d < bestd:
                                bestd = d
                                best = (r, c)
                    if best and game.is_valid_move(*best):
                        bx = board_offset_x + best[1]*CELL + best[0]*(CELL//2)
                        by = board_offset_y + best[0]*int(CELL*0.86)
                        bd = (mx-bx)**2 + (my-by)**2
                        if bd <= (CELL//2)**2:
                            hover_pos = best
                
                # Trigger redraw if hover changed
                if old_hover != hover_pos:
                    need_redraw = True
            
            if event.type==pygame.MOUSEBUTTONDOWN and event.button==1 and not game.terminal and game.turn==human:
                mx,my=event.pos
                
                # Check if click is on board (not panel)
                if mx < screen_width - PANEL_WIDTH:
                    best=None; bestd=1e9
                    for r in range(size):
                        for c in range(size):
                            x = board_offset_x + c*CELL + r*(CELL//2)
                            y = board_offset_y + r*int(CELL*0.86)
                            d=(mx-x)**2+(my-y)**2
                            if d<bestd: bestd=d; best=(r,c)
                    
                    if best and game.is_valid_move(*best):
                        bx = board_offset_x + best[1]*CELL + best[0]*(CELL//2)
                        by = board_offset_y + best[0]*int(CELL*0.86)
                        bd=(mx-bx)**2 + (my-by)**2
                        if bd <= (CELL//2)**2:
                            # Make the move
                            game.make_move(*best)
                            
                            # Sync move history to show all moves
                            sync_move_history()
                            
                            # Update evaluation and mark panel dirty
                            update_evaluation()
                            panel_dirty = True
                            draw()
                            
                            # Reset frame safety counter after move
                            frame_safety_count = 0
                            
                            if not game.terminal:
                                # Handle swap
                                if len(game.move_history)==1 and game.swap_rule and not game.swap_offered:
                                    game.offer_swap(); draw()
                                    time.sleep(0.5)
                                    if bot and hasattr(bot,'decide_swap') and bot.decide_swap(game):
                                        game.do_swap(); draw(); continue
                                
                                # Bot's turn
                                if bot:
                                    # Show "Bot thinking..." indicator
                                    ai_thinking = True
                                    draw()
                                    
                                    # Display thinking message
                                    thinking_font = pygame.font.SysFont("Arial", 32, bold=True)
                                    thinking_text = thinking_font.render("Bot thinking...", True, ACCENT_COLOR)
                                    thinking_rect = thinking_text.get_rect(center=(screen_width // 2, 50))
                                    
                                    # Draw overlay
                                    overlay = pygame.Surface((screen_width, 100))
                                    overlay.set_alpha(180)
                                    overlay.fill((0, 0, 0))
                                    screen.blit(overlay, (0, 0))
                                    screen.blit(thinking_text, thinking_rect)
                                    pygame.display.flip()
                                    
                                    # For Expert bot (NeuralMCTS), balance simulations for strong play
                                    if isinstance(bot, ExpertBot) and hasattr(bot, 'mcts_agent'):
                                        original_sims = bot.mcts_agent.simulations
                                        # Balance: enough simulations for strong play, but not too slow
                                        empty_cells = len(game.valid_moves())
                                        total_cells = game.size * game.size
                                        moves_played = total_cells - empty_cells
                                        
                                        # Increased simulations for stronger play (but still reasonable)
                                        if empty_cells < 20:  # Late endgame - critical, use more
                                            bot.mcts_agent.simulations = min(original_sims, 400)
                                        elif empty_cells < 40:  # Mid-endgame - important
                                            bot.mcts_agent.simulations = min(original_sims, 300)
                                        elif moves_played < 10:  # Early game - can use fewer
                                            bot.mcts_agent.simulations = min(original_sims, 200)
                                        else:  # Mid game - balanced
                                            bot.mcts_agent.simulations = min(original_sims, 250)
                                        
                                        print(f"[BOT] Using {bot.mcts_agent.simulations} simulations (empty cells: {empty_cells})")
                                    
                                    # Make bot move (with reduced simulations for Expert bot to prevent freezing)
                                    try:
                                        # Update thinking indicator before starting
                                        thinking_text = thinking_font.render("Bot thinking...", True, ACCENT_COLOR)
                                        overlay.fill((0, 0, 0))
                                        screen.blit(overlay, (0, 0))
                                        screen.blit(thinking_text, thinking_rect)
                                        pygame.display.flip()
                                        
                                        # Add time limit for bot move (15 seconds max)
                                        move_start_time = time.time()
                                        max_move_time = 15.0
                                        
                                        # Call bot move (this will block, but we've reduced simulations for Expert)
                                        mv = bot.select_move(game)
                                        
                                        move_time = time.time() - move_start_time
                                        if move_time > max_move_time * 0.8:  # If took >80% of max time
                                            print(f"[WARNING] Bot move took {move_time:.1f}s - consider reducing simulations further")
                                        
                                        # Restore original simulations if changed
                                        if isinstance(bot, ExpertBot) and hasattr(bot, 'mcts_agent'):
                                            bot.mcts_agent.simulations = original_sims
                                        
                                    except Exception as e:
                                        print(f"[ERROR] Bot move exception: {e}")
                                        # Fallback to simpler bot logic
                                        try:
                                            if isinstance(bot, ExpertBot):
                                                fallback = IntermediateBot()
                                                mv = fallback.select_move(game)
                                            else:
                                                valid = game.valid_moves()
                                                mv = valid[0] if valid else None
                                        except:
                                            # Last resort - just pick first valid move
                                            valid = game.valid_moves()
                                            mv = valid[0] if valid else None
                                    
                                    ai_thinking = False
                                    
                                    if mv and running:
                                        game.make_move(*mv)
                                        # Sync move history to show all moves
                                        sync_move_history()
                                        update_evaluation()
                                        panel_dirty = True
                                        draw()
                                        
                                        # Reset frame safety counter after move
                                        frame_safety_count = 0
                                    
                                    if not running:
                                        break
        
        if game.terminal:
            winner_text = "You Win!" if game.winner==human else "Bot Wins!"
            # Draw final state with winner message
            draw()
            
            # Display winner message overlay
            winner_font = pygame.font.SysFont("Arial", 72, bold=True)
            winner_color = (235, 90, 90) if game.winner==human else (90, 140, 235)
            winner_surface = winner_font.render(winner_text, True, winner_color)
            winner_rect = winner_surface.get_rect(center=(screen_width // 2, screen_height // 2))
            
            # Draw semi-transparent overlay
            overlay = pygame.Surface((screen_width, screen_height))
            overlay.set_alpha(200)
            overlay.fill((0, 0, 0))
            screen.blit(overlay, (0, 0))
            screen.blit(winner_surface, winner_rect)
            
            # Draw "Press any key to exit" message
            exit_font = pygame.font.SysFont("Arial", 32)
            exit_text = exit_font.render("Press any key or click to exit", True, (250, 250, 250))
            exit_rect = exit_text.get_rect(center=(screen_width // 2, screen_height // 2 + 100))
            screen.blit(exit_text, exit_rect)
            pygame.display.flip()
            
            # Wait for user input to exit
            waiting_for_exit = True
            while waiting_for_exit:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting_for_exit = False
                        break
                    if event.type == pygame.KEYDOWN or (event.type == pygame.MOUSEBUTTONDOWN):
                        waiting_for_exit = False
                        break
                clock.tick(60)
            
            running = False
            break
        
        clock.tick(60)
    
    pygame.quit()

# -------------------- CLI --------------------
def cli_play(human_first=True, size=BOARD_SIZE, agent=None):
    game=HexGame(size=size, swap_rule=True)
    human=HexGame.P1 if human_first else HexGame.P2
    bot=agent
    print(f"CLI Hex {size}x{size}. You are {'P1' if human==HexGame.P1 else 'P2'}")
    while not game.terminal and game.valid_moves():
        game.print_board()
        if game.turn==human:
            if len(game.move_history)==1 and game.swap_rule and not game.swap_offered:
                game.offer_swap(); ans=input("Swap? (y/n): ").strip().lower()
                if ans=='y': game.do_swap(); continue
            mv=None
            while mv is None:
                try:
                    s=input("Enter r c: "); r,c=map(int,s.strip().split())
                    if game.is_valid_move(r,c): mv=(r,c)
                    else: print("Invalid move")
                except Exception:
                    print("Parse error; example '3 4'")
            game.make_move(*mv)
        else:
            if len(game.move_history)==1 and game.swap_rule and not game.swap_offered:
                game.offer_swap();
                if bot.decide_swap(game): game.do_swap(); print("Bot swapped."); continue
            print("Bot thinking..."); mv=bot.select_move(game); print("Bot plays",mv); game.make_move(*mv)
    game.print_board(); print("Winner:", "Human" if game.winner==human else "Bot")

# -------------------- Entry --------------------
if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("PyTorch required: pip install torch"); sys.exit(1)
    if not NP_AVAILABLE:
        print("NumPy required: pip install numpy"); sys.exit(1)
    WEIGHTS='hex_net_weights_11x11_torch.pth'
    DEVICE="cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    net=TinyNet(board_size=BOARD_SIZE, hidden=256, lr=1e-3, device=DEVICE)
    if '--train' in sys.argv:
        print("\n[MODE] Training")
        net.load_weights(WEIGHTS)  # warm-start if exists
        # Visualization disabled by default for faster training (use --visualize to enable)
        visualize_training = '--visualize' in sys.argv
        if not visualize_training:
            print("[TRAIN] Visualization disabled for faster training. Use --visualize flag to enable.")
        train_selfplay(net, iterations=TRAIN_ITERS, games_per_iter=SELFPLAY_GAMES, visualize=visualize_training)
        net.save_weights(WEIGHTS)
    else:
        print("\n[MODE] Play vs AI - Select your opponent")
        net.load_weights(WEIGHTS)
        # Pass None for agent to show selection menu, and pass net for Expert bot
        pygame_gui(agent=None, size=BOARD_SIZE, net=net)
