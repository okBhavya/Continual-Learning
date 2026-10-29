# Literature Review: Continual Reinforcement Learning Under Task Similarity and Conflict

**Author:** Bhavya Gupta  
**Date:** 2026-10-29  
**Focus:** Knowledge transfer, catastrophic forgetting, negative interference, and plasticity loss in continual reinforcement learning (CRL).

---

## 1. Theoretical Foundations and the CRL Challenge

Continual Reinforcement Learning (CRL) investigates sequential decision-making where an agent encounters an ordered sequence of tasks $\mathcal{T}_1, \mathcal{T}_2, \dots, \mathcal{T}_N$, drawn from a non-stationary task distribution. Each task $\mathcal{T}_k$ is characterized by a Markov Decision Process (MDP) $\mathcal{M}_k = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}_k, \mathcal{R}_k, \gamma \rangle$.

Unlike supervised continual learning—where data distributions are static within each task and labels are provided—continual RL presents three distinct compounded failure modes:
1. **Catastrophic Forgetting (CF):** Updating the shared network parameter vector $\theta$ on the current task objective $\mathcal{L}_k(\theta)$ rapidly overwrites features critical to earlier tasks $\mathcal{T}_j$ ($j < k$), collapsing historical returns $R_{k,j} \ll R_{j,j}$.
2. **Negative Transfer & Interference:** When two consecutive tasks have conflicting dynamics or reward structures (e.g., opposite optimal actions $\pi_k^*(s) \neq \pi_j^*(s)$ for $s \in \mathcal{S}$), sharing representations or naively replaying past transitions degrades learning efficiency on the new task.
3. **Loss of Plasticity (Trainability Decay):** As a network is sequentially trained across multiple non-stationary regimes, standard gradient descent causes weight norms to inflate, activation saturation, and representation rank collapse (Neural Tangent Kernel / feature covariance rank degradation). Consequently, the agent loses the ability to adapt to subsequent tasks, regardless of whether it retained past knowledge.

---

## 2. In-Depth Analysis of Key Recent Literature

### 2.1 Self-Composing Policies for Scalable Continual Reinforcement Learning (Malagón et al., ICML 2024)
- **Publication:** International Conference on Machine Learning (ICML 2024, Oral).
- **Core Concept (CompoNet):** Rather than forcing all knowledge into a single monolithic network or training isolated disjoint sub-networks, CompoNet constructs a cascading, modular architecture. For each new task $\mathcal{T}_k$, a new module is instantiated with connections to previously learned frozen modules. The agent learns to *self-compose* past policies with new policy parameters.
- **Key Mathematical / Algorithmic Mechanism:** Total parameter growth is constrained to scale strictly linearly $\mathcal{O}(N)$ rather than quadratically. Older pathways are frozen to strictly prevent catastrophic forgetting, while forward connections enable zero-shot or few-shot forward transfer.
- **Strengths:** Eliminates catastrophic forgetting by structural isolation; enables positive forward transfer when tasks share sub-skills.
- **Limitations & Practical Constraints:**
  - Requires explicit task boundary indicators or task IDs to allocate modules.
  - Linear growth in memory footprint and forward pass computation as $N$ expands.
  - Inapplicable to pure single-head, fixed-capacity network settings where task identity is unavailable during inference.

### 2.2 Parseval Regularization for Continual Reinforcement Learning (Chung et al., NeurIPS 2024)
- **Publication:** Advances in Neural Information Processing Systems (NeurIPS 2024).
- **Core Concept:** Connects plasticity loss in continual RL to ill-conditioned weight matrices and feature rank collapse. Demonstrates that weight matrices lose full rank and diverge from isometry under non-stationary updates.
- **Key Mathematical Mechanism:** Enforces a Parseval frame condition on weight matrices $W_l$:
  $$\mathcal{L}_{\text{Parseval}}(W_l) = \frac{\beta}{2} \| W_l^T W_l - I \|_F^2$$
  where $\| \cdot \|_F$ denotes the Frobenius norm. By regularizing $W_l^T W_l \approx I$, Parseval regularization maintains layer-wise isometry, bounded Lipschitz constants, and maximal singular value preservation, directly protecting the network's capacity to continue learning new tasks.
- **Benchmarks Evaluated:** Gridworld, CARL (Contextual Action RL), Meta-World.
- **Strengths:** Task-agnostic, zero memory overhead (no replay buffer needed), directly counters representation collapse and gradient vanishing/explosion.
- **Limitations & Research Gap:** Parseval regularization primarily preserves *plasticity* (future learning ability), but on its own does not provide explicit memory retention for past tasks under high task conflict (does not prevent catastrophic forgetting of previously acquired skills).

### 2.3 Continual Reinforcement Learning by Planning with Online World Models (Liu et al., ICML 2025)
- **Publication:** International Conference on Machine Learning (ICML 2025, Spotlight).
- **Core Concept (Continual Bench & Online Agent):** Addresses the fragile nature of continual parametric policy/value networks by decoupling dynamics modeling from task-specific policy optimization. Employs a Follow-The-Leader (FTL) shallow online world model paired with Model Predictive Control (MPC) planning.
- **Key Findings:** Proven regret bounds of $\mathcal{O}(\sqrt{K^2 D \log T})$ under unified world dynamics. Because the dynamics model is updated online and task policies are planned on the fly using task-specific reward definitions, the agent sidesteps catastrophic policy forgetting.
- **Strengths:** Strong theoretical guarantees; avoids value-function approximation churn.
- **Limitations & Practical Constraints:**
  - Requires planning-time computation (MPC rollouts at every timestep), which is orders of magnitude slower than feedforward policy evaluation.
  - Requires accurate knowledge of reward functions for each task, or learning separate reward heads.
  - Highly dependent on the world dynamics being unified or smoothly evolving.

### 2.4 Mitigating Plasticity Loss in Continual Reinforcement Learning by Reducing Churn (Tang et al., ICML 2025)
- **Publication:** International Conference on Machine Learning (ICML 2025).
- **Core Concept (C-CHAIN):** Investigates *churn*—defined as the variance in network predictions on out-of-batch or historical states caused by single mini-batch gradient steps on current data.
- **Key Insight:** Excessive churn causes rapid degradation of the Neural Tangent Kernel (NTK) rank, triggering catastrophic plasticity loss. By constraining prediction churn across anchor states, the network retains representation stability without freezing parameters.
- **Benchmarks Evaluated:** Classic Control, ProcGen, DeepMind Control Suite, MinAtar.
- **Strengths:** Targets the root mechanism of optimization degradation; works in continuous or discrete action spaces.
- **Limitations:** Anchor state selection requires either cached inputs or auxiliary data buffers; tuning churn regularization strength $\lambda_{\text{churn}}$ is sensitive to task transition magnitude.

---

## 3. Summary Comparison of Modern CRL Approaches

| Method | Venue | Primary Mechanism | Memory Overhead | Protects Stability (CF)? | Protects Plasticity? | Operates Without Task ID? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CompoNet** | ICML 2024 | Dynamic modular expansion & self-composition | $\mathcal{O}(N)$ weights | Yes (via module freezing) | Yes | No (requires task index) |
| **Parseval Reg.** | NeurIPS 2024 | Weight matrix orthogonality penalty ($\|W^TW - I\|_F^2$) | None ($\mathcal{O}(1)$) | Partial (stabilizes gradients) | Strong | Yes |
| **Online World Models** | ICML 2025 | FTL online dynamics + MPC planning | Dynamics model only | Yes (for shared dynamics) | Yes | Yes (if rewards known) |
| **C-CHAIN** | ICML 2025 | Churn reduction on anchor states | Anchor states buffer | Moderate | Strong | Yes |
| **Elastic Weight Cons. (EWC)** | PNAS 2017 | Fisher-weighted quadratic parameter penalty | $\mathcal{O}(P)$ Fisher info | Weak in RL (scale shift) | Poor (freezes weights) | Typically yes |
| **Experience Replay (ER)** | Classic | Uniform rehearsal from past task memory | $\mathcal{O}(M)$ transitions | Moderate | Moderate | Yes |

---

## 4. The Research Gap

A critical question remains unresolved across these paradigms:
> **How should an agent with fixed neural capacity and no task IDs balance retention and adaptation when the similarity between tasks is unknown and non-stationary?**

Under **high task similarity** (e.g., slight physical perturbations), past knowledge should be leveraged for forward transfer, and heavy regularization or aggressive replay of outdated trajectories is both wasteful and restrictive.

Under **high task conflict / dissimilarity** (e.g., inverted action semantics or conflicting control directions), naive rehearsal forces the network to fit contradictory targets $(s, a) \mapsto Q(s,a)$, causing policy dithering, severe negative transfer, and destructive gradient churn. Furthermore, rigid parameter penalties like EWC over-constrain the shared representations, destroying plasticity without effectively salvaging performance.

This motivates an **adaptive, conflict-aware mechanism** that:
1. Dynamically measures policy/gradient compatibility or representation divergence between past and current tasks.
2. Modulates replay allocation and regularization strength adaptively: encouraging forward transfer when compatibility is positive, while shielding past memory and maintaining weight conditioning (via Parseval frame regularization) when conflict is detected.
