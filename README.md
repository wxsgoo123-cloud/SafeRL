# 🧠 Safe Learning Through Controlled Expansion of Exploration Set

Official implementation of the numerical experiments in:



This repository contains the **numerical experiments** for the proposed algorithm **LearnSEES**, a **model-free safe reinforcement learning** method that guarantees *almost-sure safety* by progressively expanding a verified **safe exploration set** using **Gaussian Processes (GPs)** or **neural uncertainty models**.

---

## 🚀 Overview

LearnSEES addresses reinforcement learning problems where the **safety of state–action pairs is unknown a priori**.  
The method ensures that the **entire training process** remains within a *safe set* that expands cautiously based on predictive uncertainty.

At each episode:

1. **Prediction** – Update GP/NN model of the cost function.  
2. **Exploration Set Expansion** – Define the safe region via GP confidence bounds.  
3. **Policy Update** – Run Q-learning constrained to the current safe region.  
4. **Deployment** – Collect new data safely to refine the model.

The algorithm guarantees both:
- 📉 *Asymptotic convergence* to the optimal safe policy  
- 📈 *Finite-episode online regret bound*

- 
### 🧩 Description of Tasks

| Task | Type | Description | Paper Section |
|------|------|--------------|----------------|
| **Gridworld (Discrete)** | Tabular RL | 12×12 grid where the agent must reach a goal while avoiding risky regions. Costs depend on cell color. Compares LearnSEES vs. Linear Search baseline under various safety thresholds (`ϵrisk ∈ {0.25, 0.75, 1.25}`). | Sec. 5.1 |
| **Gridworld (Continuous)** | Continuous control | 10×10 continuous environment with two circular “danger zones.” Agent learns to reach a goal region using continuous actions. Uses GP with discretized predictions for safe expansion. | Sec. 5.2 |
| **CartPole** | Classic control | Balancing a pole on a cart with per-step safety constraint based on position and angle. Compares LearnSEES vs. DQN and ActSafe. Shows lower violation rates and safe convergence. | Sec. 5.3 |
| **Safety-Gymnasium (PointGoal1-v0)** | High-dimensional continuous RL | Point robot navigates toward a moving goal while avoiding hazards. Uses neural uncertainty model instead of GP. Compared with Constrained Policy Optimization (CPO). | Sec. 5.4 |

Each task demonstrates the algorithm’s ability to **maintain safety during training** while converging to **near-optimal policies**, covering both **discrete and continuous** environments.

---


