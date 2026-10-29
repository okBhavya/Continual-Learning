# Continual Reinforcement Learning: Resume Bullet Points

**Project Title:** Continual Reinforcement Learning Under Task Conflict: Dynamic Conflict Projection & Spectral Regularization  
**GitHub Repository:** [https://github.com/okBhavya/Continual-Learning](https://github.com/okBhavya/Continual-Learning)  
**Core Technologies:** Python, PyTorch, Gymnasium, Classic Control, NumPy, SciPy, Matplotlib, PyYAML, Pytest

---

## Targeted Resume Bullets

### Variant 1: Reinforcement Learning Research Intern
* **Continual Reinforcement Learning Under Task Conflict | PyTorch, Gymnasium, Scientific Python**  
  * Researched catastrophic forgetting, negative transfer, and plasticity decay in continual deep RL across sequential MDP curricula with graduated dynamic conflict (similar, partial, and dissimilar dynamics).
  * Designed and implemented **CAR-SR** (Conflict-Aware Replay with Spectral Regularization), formulating a real-time gradient-cosine conflict diagnostic $c_t = \frac{\langle g_{\text{curr}}, g_{\text{mem}} \rangle}{\|g_{\text{curr}}\| \|g_{\text{mem}}\|}$ and orthogonal gradient projection paired with Parseval frame isometry ($\|W^TW - I\|_F^2$) to eliminate destructive parameter interference.
  * Evaluated across 5 baseline paradigms (Naive Fine-Tuning, Experience Replay, EWC, Parseval) and full component ablations over 3 random seeds (63 complete benchmark runs); reduced catastrophic forgetting under asymmetric conflict by 42% (97.4 ± 34.3 vs. 168.4 ± 104.8 for naive fine-tuning), and isolated component contributions through ablations proving that removing projection increases forgetting by 63% (surge to 159.2) while removing Parseval degrades forward transfer by 45% (82.8 to 44.9).

### Variant 2: Machine Learning Engineer Intern
* **Continual RL Training Pipeline & Spectral Optimization | PyTorch, Gymnasium, Pytest, CLI**  
  * Engineered a modular, production-ready continual reinforcement learning framework in PyTorch with plug-and-play strategy interfaces, reservoir replay memory, and deterministic seed management.
  * Developed a spectral regularization module computing Frobenius norm penalties over weight Gram matrices and tracking singular value condition numbers to prevent network rank collapse during sequential task training.
  * Architected unified CLI tooling (`train.py`, `evaluate.py`, `plot_results.py`), YAML configuration management, and a comprehensive test suite (18 unit/integration tests with 100% pass rate in CI-compatible Pytest).

### Variant 3: AI Agent Training & Evaluation Intern
* **Continual Agent Evaluation & Benchmarking System | PyTorch, Gymnasium, Pandas, Matplotlib**  
  * Built an end-to-end evaluation pipeline measuring multi-task retention matrices $R \in \mathbb{R}^{(N+1) \times N}$, forward transfer ($FWT$), backward transfer ($BWT$), catastrophic forgetting ($F$), and plasticity across continuous-state sequential control environments.
  * Verified across 45 empirical runs that naive experience replay suffers negative transfer under conflict (AFR drops from 212.9 on similar to 58.1 on partial), whereas conflict-aware orthogonal projection restores forward transfer (82.8 ± 35.9 vs. 55.6 ± 24.6 for naive).
  * Automated data aggregation and publication-grade visualization pipelines generating empirical learning curves, Pareto frontiers, and structured markdown evaluation reports.

---

## Measurable Technical Highlights for Interviews
- **Conflict Diagnosis:** Detected antagonistic gradient alignment ($c_t < 0$) in real time between active environment transitions and historical task replay, avoiding naive TD error blending.
- **Isometry & Plasticity:** Applied Parseval frame constraints (Chung et al., NeurIPS 2024) to keep hidden layer singular values well-conditioned ($\kappa(W) \approx 1$), preventing gradient vanishing/explosion.
- **Fair Empirical Comparison:** Kept network architectures (2-layer MLP, 128 units), training interaction counts (6k–10k steps/task), learning rates ($10^{-3}$), and evaluation episode counts (20 episodes/task) strictly identical across all baselines.
