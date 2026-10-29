# Research Plan: Conflict-Aware Replay with Spectral Regularization (CAR-SR)

**Author:** Bhavya Gupta  
**Date:** 2026-10-29  
**Status:** Working Specification & Experimental Protocol  

---

## 1. Problem Formulation & Motivation

In continual reinforcement learning (CRL), an agent is trained on a sequential curriculum $\mathcal{T}_1 \to \mathcal{T}_2 \to \dots \to \mathcal{T}_N$ under a single neural network architecture with no external task oracle or task ID provided at test time.

Prior investigations (including our preliminary course findings in `RL_MidTerm.pdf` and recent literature from ICML 2024–2025 and NeurIPS 2024) demonstrate that:
1. **Under High Task Similarity:** Tasks share invariant dynamics and state-action mappings. Naive fine-tuning retains performance moderately well, but static parameter regularization (like EWC) heavily penalizes weight adaptation, inducing premature plasticity loss (e.g., EWC final return dropped from ~304 to 140 on similar sequences).
2. **Under Task Conflict (Partial or Complete Action Flips):** Tasks demand divergent or opposing policies $\pi^*(s)$ in overlapping regions of the state space. 
   - **Naive fine-tuning** completely overwrites past skills ($337 \to 29 \to 9.7$ score on Task 1).
   - **Naive Experience Replay (ER)** uniformly blends past transitions into the mini-batch at a fixed ratio $\alpha = 0.5$. When past and current gradients conflict ($\langle \nabla \mathcal{L}_{\text{curr}}, \nabla \mathcal{L}_{\text{past}} \rangle < 0$), the network is forced to minimize mutually contradictory Bellman targets, creating severe gradient churn and high seed variance (e.g. seeds ending at 25 vs 497).
   - **EWC** estimates diagonal Fisher information on DQN via softmax policy approximations, but Fisher values overwhelmingly concentrate on the final linear layer (Slide 15 diagnostic: output layer $\approx 1.0$, hidden layers $\approx 0.001$). This rigidly freezes action selection while failing to preserve hidden representation rank.

---

## 2. Hypothesis

> **Central Hypothesis:**  
> Explicitly diagnosing task conflict via mini-batch gradient inner products and dynamically mitigating gradient opposition—combined with Parseval frame spectral regularization on hidden layers—will resolve the stability–plasticity dilemma across diverse task similarity regimes:
> 1. In **concordant regimes** ($\langle g_{\text{curr}}, g_{\text{past}} \rangle \ge 0$), it preserves high plasticity and facilitates forward transfer without unnecessary weight freezing.
> 2. In **conflicting regimes** ($\langle g_{\text{curr}}, g_{\text{past}} \rangle < 0$), it prevents destructive interference by projecting conflicting gradients while maintaining hidden layer isometry, significantly reducing catastrophic forgetting without sacrificing adaptation to the new task.

---

## 3. Mathematical Formulation: The CAR-SR Method

Let $\theta$ denote the parameters of the Q-network $Q_\theta(s, a)$. Training proceeds in sequential task phases.

### 3.1 Dual-Stream Loss Formulation
At training step $t$, the agent samples:
- Current-task mini-batch $\mathcal{B}_{\text{curr}} \sim \mathcal{D}_{\text{curr}}$, producing TD loss $\mathcal{L}_{\text{curr}}(\theta)$.
- Past-task rehearsal mini-batch $\mathcal{B}_{\text{mem}} \sim \mathcal{D}_{\text{mem}}$, producing replay loss $\mathcal{L}_{\text{mem}}(\theta)$.

Let the respective parameter gradients be:
$$g_{\text{curr}} = \nabla_\theta \mathcal{L}_{\text{curr}}(\theta), \quad g_{\text{mem}} = \nabla_\theta \mathcal{L}_{\text{mem}}(\theta)$$

### 3.2 Gradient Conflict Diagnosis
We define the instantaneous cosine gradient alignment:
$$c_t = \frac{\langle g_{\text{curr}}, g_{\text{mem}} \rangle}{\|g_{\text{curr}}\|_2 \|g_{\text{mem}}\|_2 + \epsilon}$$

- **Case 1 ($c_t \ge 0$ - Concordant / Positive Transfer):**  
  Current learning aligns with or is orthogonal to past memory. The standard combined gradient is retained:
  $$g_{\text{task}} = g_{\text{curr}} + \lambda_{\text{mem}} g_{\text{mem}}$$

- **Case 2 ($c_t < 0$ - Conflicting / Destructive Interference):**  
  The current update directly degrades past policy performance. Following projection lifelong learning principles adapted to value-based RL, we project $g_{\text{curr}}$ onto the normal plane of $g_{\text{mem}}$:
  $$g_{\text{curr}}^{\text{proj}} = g_{\text{curr}} - \frac{\langle g_{\text{curr}}, g_{\text{mem}} \rangle}{\|g_{\text{mem}}\|_2^2 + \epsilon} g_{\text{mem}}$$
  and form the conflict-resolved gradient:
  $$g_{\text{task}} = g_{\text{curr}}^{\text{proj}} + \lambda_{\text{mem}} g_{\text{mem}}$$
  This guarantees that $\langle g_{\text{task}}, g_{\text{mem}} \rangle = \lambda_{\text{mem}} \|g_{\text{mem}}\|_2^2 \ge 0$, strictly preventing updates on the current task from increasing past-memory loss.

### 3.3 Parseval Frame Regularization for Plasticity Preservation
Following Chung et al. (NeurIPS 2024), we combat representation rank collapse and plasticity loss across sequential tasks by penalizing deviation from semi-orthogonality in the hidden layer weight matrices $W_l$:
$$\mathcal{L}_{\text{Parseval}}(\theta) = \frac{\beta}{2} \sum_{l=1}^{L-1} \| W_l^T W_l - I \|_F^2$$
where $\|\cdot\|_F$ is the Frobenius norm.

The total optimization objective satisfies:
$$g_{\text{total}} = g_{\text{task}} + \beta \nabla_\theta \mathcal{L}_{\text{Parseval}}(\theta)$$

---

## 4. Key Differences from Existing Methods

| Feature | Naive Fine-Tuning | Reservoir Replay | EWC (Kirkpatrick 2017) | Parseval Only (Chung 2024) | Proposed CAR-SR |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Old-Task Memory** | None (wiped) | Uniform reservoir | None (Fisher penalty) | None | Bounded reservoir |
| **Conflict Detection**| None | None | None | None | Real-time gradient cosine $c_t$ |
| **Interference Handling**| Overwrites | Averages blindly | Freezes weights | None | Projected non-interfering update |
| **Plasticity Defense**| None | Implicit | None (destroys plasticity)| Orthogonality penalty | Orthogonality penalty |
| **Task ID Requirement**| No | No | No | No | No |

---

## 5. Experimental Protocol & Hypotheses Testing

### 5.1 Benchmark Environments
We evaluate on the three graduated conflict sequences established in `envs.py`:
1. **`similar` sequence:** Moderate physics variations ($\Delta$ length, gravity, force). Minimal policy conflict.
2. **`partial` sequence:** Physics change followed by `FlipWhenRight` (actions inverted strictly on $x > 0$). Conflict localized to half the state space; resolvable by a single network with sufficient capacity and plasticity.
3. **`dissimilar` sequence:** Physics change followed by `FlipActions` (full action inversion). Direct global policy conflict; serves as the ultimate stress test.

### 5.2 Baselines for Fair Comparison
All methods will be trained on identical neural architectures (2-layer MLP, 128 hidden units per layer, ReLU), identical training budgets (10,000 steps per task), identical batch sizes (64), learning rates ($10^{-3}$), and evaluation protocols (20 test episodes per task at each evaluation stage) across 3 distinct random seeds.

1. **Sequential Fine-Tuning (`naive`):** Standard lower bound for catastrophic forgetting.
2. **Experience Replay (`replay`):** Fixed 50% replay mixing from past task buffers (standard CRL rehearsal baseline).
3. **Elastic Weight Consolidation (`ewc`):** Fisher-weighted parameter regularization.
4. **Parseval Regularization (`parseval`):** Standalone spectral regularization baseline (NeurIPS 2024).
5. **Proposed Method (`car_sr`):** Conflict-Aware Replay + Parseval Orthogonality.

### 5.3 Formal Evaluation Metrics
Following standard continual learning formulations:
- **Return Matrix $R \in \mathbb{R}^{(N+1) \times N}$:**
  $R_{i,j}$ represents the average evaluation return on task $j$ after completing training on task $i$. ($R_{0,j}$ is the untrained initial policy).
- **Average Final Return ($\text{AFR}$):**
  $$\text{AFR} = \frac{1}{N} \sum_{j=1}^N R_{N,j}$$
- **Catastrophic Forgetting ($F$):**
  $$F = \frac{1}{N-1} \sum_{j=1}^{N-1} \left( \max_{t \in \{j, \dots, N-1\}} R_{t,j} - R_{N,j} \right)$$
- **Backward Transfer ($BWT$):**
  $$BWT = \frac{1}{N-1} \sum_{j=1}^{N-1} (R_{N,j} - R_{j,j})$$
- **Forward Transfer ($FWT$):**
  $$FWT = \frac{1}{N-1} \sum_{j=2}^N (R_{j-1,j} - R_{0,j})$$
- **Plasticity ($P$):**
  $$P = \frac{1}{N} \sum_{j=1}^N R_{j,j}$$

### 5.4 Potential Failure Modes & How to Disprove the Hypothesis
- **Disproving Condition 1:** If in the `partial` or `dissimilar` sequence, CAR-SR performs no better or worse than naive replay and EWC in Catastrophic Forgetting ($F$) and Final Return ($\text{AFR}$).
- **Disproving Condition 2:** If Parseval regularization overly restricts the dynamic range of the network, degrading plasticity on new tasks ($P_{\text{CAR-SR}} \ll P_{\text{naive}}$).
- **Disproving Condition 3:** If gradient projection computational overhead degrades step throughput without measurable empirical gain over simpler ablations (e.g., Replay + Parseval alone).
