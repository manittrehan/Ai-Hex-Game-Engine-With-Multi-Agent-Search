# 11×11 Hex Game Engine & Multi-Agent AI Suite

A high-performance, object-oriented 11×11 Hex game engine and AI suite built from scratch in Python. This project combines a fully interactive graphical interface with a diverse range of competitive game-playing agents, bridging the gap between classical heuristic tree searches and modern deep reinforcement learning architectures.

---

## 📌 Project Overview

Hex is a deterministic, perfect-information strategy game played on an $11 \times 11$ rhombic grid of hexagons. Players (Red vs. Blue) alternate placing markers on empty spaces, aiming to form an unbroken chain connecting their respective opposing sides of the board. Because Hex cannot end in a tie, the game guarantees a win for one player, making it an exceptionally rich benchmark for artificial intelligence.

This repository provides a complete sandbox to play, train, and benchmark different AI methodologies against one another:
* **Deep Reinforcement Learning:** An AlphaZero-inspired pipeline combining Convolutional Residual Networks with PUCT-driven Monte Carlo Tree Search.
* **Heuristic Search:** An Alpha-Beta Pruning minimax agent driven by graph-theory shortest-path algorithms.
* **Traditional Look-ahead:** A vanilla MCTS agent using dynamic rollouts.

---

## 🚀 Key Features

### 1. High-Fidelity Game Engine
* **Union-Find Win Detection:** Utilizes a Disjoint-Set Data Structure (Union-Find) with path compression to evaluate winning connections in $\mathcal{O}(\alpha(N))$ nearly constant time, bypassing slow global board scans.
* **Pie Rule (Swap Rule):** Fully supports the tournament standard "swap rule" to mitigate the absolute first-player winning advantage.

### 2. Deep RL Agent (AlphaGo Zero Style)
* **Dual-Headed Policy-Value ResNet:** Built using **PyTorch**, featuring deep 2D convolutional networks with residual blocks. The network ingests the current board state tensor and concurrently outputs a probability distribution over moves (Policy Head) and a positional win probability scalar (Value Head).
* **PUCT-Guided MCTS:** Replaces standard random playouts with a bounded Polynomial Upper Confidence Trees search, using the neural network’s prior probabilities to intelligently prune unpromising branches.

### 3. Classical & Heuristic AI
* **Alpha-Beta Pruning Minimax:** Evaluates non-terminal positions using a custom-engineered graph-theoretic heuristic. It calculates the minimum "virtual connections" remaining for both sides using a modified Dijkstra’s pathfinding algorithm.
* **Endgame Solver:** Automatically transitions the Alpha-Beta agent into an exact exhaustive search when the remaining empty cell count falls below a critical threshold.
* **Adaptive Simulation MCTS:** Uses vanilla Monte Carlo Tree Search, utilizing dynamic simulation counts that scale based on the urgency/complexity of the position.

### 4. Interactive GUI & Benchmarking
* **Pygame Visualizer:** A clean graphical interface showcasing real-time board evaluations, current search paths, and win-probability bars.
* **Matchmaking Framework:** Command-line pipeline allowing any matchup pairing: Human vs. AI, AI vs. AI, or automated batch tournaments for Elo evaluation.

---

## 🏗️ System Architecture

The project is strictly modular, separating the game logic, rendering components, and decision-making brains:
