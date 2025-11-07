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


