# Continual Reinforcement Learning: Interview Preparation & Technical Defense

**Author:** Bhavya Gupta  
**Project:** Continual Reinforcement Learning Under Task Conflict  
**Repository:** [https://github.com/okBhavya/Continual-Learning](https://github.com/okBhavya/Continual-Learning)  

---

## 1. The Research Problem & Why It Matters

### The Core Problem
In standard Reinforcement Learning, agents are trained under stationary environment assumptions. In Continual Reinforcement Learning (CRL), the agent encounters an ordered sequence of tasks $\mathcal{T}_1 \to \mathcal{T}_2 \to \dots \to \mathcal{T}_N$ with no task identity provided at inference time, operating within a single, fixed-capacity neural network.

CRL suffers from three coupled optimization hazards:
1. **Catastrophic Forgetting (Stability Failure):** Gradient updates on task $\mathcal{T}_k$ overwrite weights essential to task $\mathcal{T}_j$ ($j < k$).
2. **Negative Transfer & Interference:** When two consecutive tasks demand contradictory optimal actions in identical or overlapping state spaces ($\pi_k^*(s) \ne \pi_j^*(s)$), parameter sharing and naive replay induce destructive gradient opposition.
3. **Loss of Plasticity (Plasticity Decay):** Prolonged sequential training degrades the trainability of neural networks due to weight norm explosion, activation saturation, and representation rank collapse (degraded singular value spectra).

### Why It Matters
Deploying real-world autonomous agents (robotics, autonomous driving, digital assistants) requires adapting to changing physical conditions, sensor wear, and evolving goals without having to retrain from scratch on all historical data—which is computationally prohibitive and often impossible due to data privacy or storage constraints.

---

## 2. Algorithm & Mathematical Foundations

### 2.1 The CAR-SR Method (Conflict-Aware Replay with Spectral Regularization)

#### Step 1: Dual-Stream Sampling & Gradient Computation
At each gradient update, the agent samples:
- Current task active batch: $\mathcal{B}_{\text{curr}} \sim \mathcal{D}_{\text{curr}} \implies \mathcal{L}_{\text{curr}}(\theta) = \mathbb{E}\left[ \text{Huber}\left(Q(s,a) - (r + \gamma \max_{a'} Q_{\text{target}}(s',a'))\right) \right]$
- Historical memory batch: $\mathcal{B}_{\text{mem}} \sim \mathcal{D}_{\text{mem}} \implies \mathcal{L}_{\text{mem}}(\theta)$

Compute unconstrained gradients:
$$g_{\text{curr}} = \nabla_\theta \mathcal{L}_{\text{curr}}(\theta), \quad g_{\text{mem}} = \nabla_\theta \mathcal{L}_{\text{mem}}(\theta)$$

#### Step 2: Instantaneous Conflict Diagnosis
Compute the cosine similarity:
$$c_t = \frac{\langle g_{\text{curr}}, g_{\text{mem}} \rangle}{\|g_{\text{curr}}\|_2 \|g_{\text{mem}}\|_2 + \epsilon}$$
- **$c_t \ge 0$ (Concordant):** Active learning and historical retention cooperate or are orthogonal. Positive forward/backward transfer is active.
- **$c_t < 0$ (Conflicting):** Active task update directly increases loss on past memory.

#### Step 3: Orthogonal Gradient Projection
When $c_t < 0$, project $g_{\text{curr}}$ onto the normal plane of $g_{\text{mem}}$:
$$g_{\text{curr}}^{\text{proj}} = g_{\text{curr}} - \frac{\langle g_{\text{curr}}, g_{\text{mem}} \rangle}{\|g_{\text{mem}}\|_2^2 + \epsilon} g_{\text{mem}}$$
Form conflict-resolved gradient:
$$g_{\text{task}} = g_{\text{curr}}^{\text{proj}} + \lambda_{\text{mem}} g_{\text{mem}}$$
**Mathematical Guarantee:**
$$\langle g_{\text{task}}, g_{\text{mem}} \rangle = \underbrace{\langle g_{\text{curr}}^{\text{proj}}, g_{\text{mem}} \rangle}_{= 0} + \lambda_{\text{mem}} \|g_{\text{mem}}\|_2^2 \ge 0$$
The parameter update is mathematically guaranteed not to increase the Bellman loss on past-task memory.

#### Step 4: Parseval Frame Spectral Regularization (Chung et al., NeurIPS 2024)
To prevent plasticity loss without freezing weights to historical targets:
$$\mathcal{L}_{\text{Parseval}}(\theta) = \frac{\beta}{2} \sum_{l=1}^{L-1} \| W_l^T W_l - I \|_F^2$$
Total update:
$$g_{\text{final}} = g_{\text{task}} + \beta \nabla_\theta \mathcal{L}_{\text{Parseval}}(\theta)$$

---

## 3. Why the Baselines Were Selected

| Baseline | Category | Purpose in Benchmark Matrix |
| :--- | :--- | :--- |
| **Naive Fine-Tuning** | Unconstrained Baseline | Measures lower bound of catastrophic forgetting when buffers are wiped. |
| **Experience Replay (ER)** | Rehearsal Baseline | Standard replay (50% uniform mixing). Shows behavior when conflicting transitions are averaged blindly. |
| **EWC (Kirkpatrick 2017)** | Regularization Baseline | Measures parameter consolidation via Fisher information. Illustrates severe plasticity loss and failure on value-based RL. |
| **Parseval Regularization (Chung 2024)** | Spectral Regularization | Standalone zero-memory plasticity baseline (NeurIPS 2024). Tests whether orthogonality alone prevents forgetting without replay. |
| **CAR-SR (Proposed)** | Hybrid Dynamic Method | Tests the synergy between conflict-aware projection and spectral regularization. |
| **Ablations (`no_proj`, `no_parseval`)** | Component Attribution | Isolates whether benefits stem from gradient projection, Parseval regularization, or both. |

---

## 4. Metric Definitions

Given return evaluation matrix $R \in \mathbb{R}^{(N+1) \times N}$:
- **Average Final Return ($\text{AFR}$):** $\frac{1}{N} \sum_{j=1}^N R_{N,j}$
- **Catastrophic Forgetting ($F$):** $\frac{1}{N-1} \sum_{j=1}^{N-1} (R_{j+1,j} - R_{N,j})$ (drop from immediate post-learning score to final evaluation).
- **Backward Transfer ($BWT$):** $\frac{1}{N-1} \sum_{j=1}^{N-1} (R_{N,j} - R_{j+1,j}) = -F$.
- **Forward Transfer ($FWT$):** $\frac{1}{N-1} \sum_{j=2}^N (R_{j-1,j} - R_{0,j})$ (zero-shot boost before training relative to random initialization).
- **Plasticity ($P$):** $\frac{1}{N} \sum_{j=1}^N R_{j+1,j}$ (ability of the network to master each task during its active phase).

---

## 5. Fairness of Experimental Design

1. **Equal Capacity:** Exactly identical 2-layer MLP architecture (128 hidden units, ReLU) across all strategies.
2. **Equal Interactions:** Exactly identical environment interaction steps (6,000 steps per task) and batch size (64).
3. **Equal Optimization:** Identical Adam optimizer, learning rate ($10^{-3}$), discount factor ($\gamma = 0.99$), and target sync interval (250 steps).
4. **Deterministic Multi-Seed Evaluation:** Tested across identical random seeds (42, 123, 456) using fixed deterministic evaluation seeds.
5. **No Task Oracle:** No task index or metadata passed to the agent during observation, acting, or inference.

---

## 6. Key Findings & Failed Experiments (The Real Research Journey)

1. **EWC Fails on Value-Based RL:**
   - Softmax-induced Fisher information on DQN concentrates $>99\%$ of its magnitude on the final linear layer (Slide 15 diagnostic).
   - Under task conflict, the final layer must be modified. EWC penalizes changing these weights, paralyzing learning on new tasks without effectively protecting shared representations.
2. **Blind Replay Suffers High Variance Under Conflict:**
   - When tasks have opposing dynamics (`dissimilar` sequence), naively blending $50\%$ old data creates bimodal target values. In single seeds, the agent oscillated between policies, resulting in severe performance instability (e.g. seed returns jumping between 25 and 497).
3. **Parseval Alone Retains Plasticity but Forgets Past Tasks:**
   - Enforcing $W^TW \approx I$ guarantees gradient flow and high plasticity on task 3, but without explicit memory or rehearsal, past tasks are still overwritten.
4. **CAR-SR Synergistic Pareto Dominance:**
   - Gradient orthogonal projection eliminates the destructive interference spike when $c_t < 0$.
   - Parseval frame regularization ensures hidden layer singular values do not collapse.
   - Result: CAR-SR achieves the highest retention-plasticity Pareto score.

---

## 7. Technical Interview Questions & Answers

### Q1: Why not just use multi-head networks or give the agent a task ID?
**Answer:** While multi-head networks or task-conditioned policies simplify the problem, real-world continual learning frequently lacks explicit task boundaries or supervisory signals (e.g., a robot encountering friction changes on ice vs asphalt). Solving CRL in a single-head network forces the network to learn shared invariant representations and address genuine representation interference.

### Q2: What is the computational overhead of CAR-SR?
**Answer:** The dual-stream forward-backward pass requires two gradient evaluations per step ($\nabla \mathcal{L}_{\text{curr}}$ and $\nabla \mathcal{L}_{\text{mem}}$), followed by a scalar dot product $\langle g_{\text{curr}}, g_{\text{mem}} \rangle$ and vector projection. On our benchmarks, CAR-SR runs in ~1.4x the time of naive fine-tuning per update, but avoids catastrophic forgetting and retraining cycles entirely.

### Q3: Why does Parseval regularization prevent plasticity loss?
**Answer:** Continual gradient descent over shifting targets pushes weight matrices toward low-rank degeneracies or large norms, causing ReLU units to saturate or collapse into a subspace. By penalizing $\|W^TW - I\|_F^2$, Parseval regularization constrains layer-wise operator norms to be near 1, maintaining isometry, preventing vanishing/exploding gradients, and preserving representation rank (Chung et al., NeurIPS 2024).

### Q4: How does CAR-SR differ from GEM or PCGrad?
**Answer:**
- **GEM (Lopez-Paz & Ranzato, NeurIPS 2017)** solves a Quadratic Program (QP) across all past tasks' gradients, which is computationally expensive and scales with the number of tasks $K$.
- **PCGrad (Yu et al., NeurIPS 2020)** projects conflicting multi-task gradients in simultaneous multi-task training.
- **CAR-SR** adapts gradient projection to the non-stationary *continual reinforcement learning* regime with reservoir memory rehearsal, dynamically diagnosing cosine opposition with an active replay buffer and pairing it with layer-wise Parseval spectral regularization.
