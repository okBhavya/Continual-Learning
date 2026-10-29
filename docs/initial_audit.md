# Starting State Audit & Baseline Assessment

**Date:** 2026-10-29  
**Repository:** `https://github.com/okBhavya/Continual-Learning`  
**Inspected Source Directories:**  
- `C:\Users\Lenovo\Downloads\Continual RL` (existing prototype codebase and midterm presentation)  
- `C:\Users\Lenovo\Downloads\Continual Reinforcement Learning` (active project root and Git working directory)  

---

## 1. Environment & Hardware Discovery

| Component | Detected Specification | Implications & Constraints |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 (AMD64) | PowerShell / cmd shell execution; standard Windows paths. |
| **System Python** | Python 3.14.2 (CPython) | CPython 3.14 lacks pre-built wheels for PyTorch and several scientific ML packages. |
| **Environment Tooling**| `uv` 0.12.23 installed | Enables rapid creation of managed CPython 3.11/3.12 virtual environments (`.venv`) with binary wheel compatibility. |
| **Project Virtualenv**| CPython 3.11.17 (`.venv\Scripts\python.exe`) | Modern, stable runtime for PyTorch 2.x, Gymnasium, Matplotlib, SciPy, PyYAML, Pytest. |
| **Processor (CPU)** | Intel(R) Core(TM) Ultra 5 125H (14 cores, 18 logical threads) | High multi-core throughput suitable for parallel simulation, multi-seed RL evaluation, and fast CPU-based neural network forward/backward passes. |
| **System Memory** | 16.0 GB Total RAM (~2.3 GB free prior to venv setup) | Lightweight memory footprint required; avoid massive uncompressed replay caches. Keep replay buffers bounded. |
| **Graphics (GPU)** | Intel(R) Arc(TM) Graphics (integrated/discrete) | No CUDA-capable NVIDIA GPU detected (`nvidia-smi` unavailable). All RL network architectures and training loops must be strictly optimized for CPU training. |

---

## 2. Remote Git Repository Audit

- **URL:** `https://github.com/okBhavya/Continual-Learning`
- **Current HEAD on Remote:** `aa232ac163465e06975ad033d02c65a96e709fa6`
- **Remote Branches:** `main`
- **Existing Remote Content:** Minimal initial commit (single `README.md` containing `# Continual-Learning`).
- **Local Git Identity:**
  - `user.name`: `Bhavya_Gupta`
  - `user.email`: `f20230494@pilani.bits-pilani.ac.in`
- **Safety Policy:** All new commits must be cleanly integrated onto `main` preserving commit lineage; `.gitignore` must strictly exclude `.venv/`, `.gemini/`, cached `.pyc`, and large binaries.

---

## 3. Existing Local Codebase Inspection (`Continual RL`)

The existing folder `C:\Users\Lenovo\Downloads\Continual RL` was inspected file by file. It represents a course midterm project ("CS F317 - Reinforcement Learning Midterm: Continual Reinforcement Learning under Task Conflict" by Bhavya Gupta and Parth Kumar Garg).

### Files Inspected:

1. **`RL_MidTerm.pdf`**:
   - 17-slide presentation outlining initial experiments on Continual RL under task conflict.
   - Core research question: *How does forgetting change as task conflict increases, and how do replay and EWC hold up?*
   - Defines three conflict levels in CartPole:
     - `similar`: Small physics perturbations ($\Delta$ pole length, gravity, force).
     - `partial`: Asymmetric conflict (actions swapped only on the right half of the track $x > 0$).
     - `dissimilar`: Inverted dynamics (full action swap $a \leftarrow 1-a$, direct contradiction).
   - Core empirical midterm findings:
     - Naive fine-tuning forgets catastrophically as conflict escalates ($337 \to 29 \to 9.7$ score on Task 1).
     - Replay helps inconsistently; high seed variance.
     - EWC struggles with value-based DQN (Fisher information concentrated on output heads, causing severe plasticity loss on new tasks).
     - Midterm planned next steps: direct conflict estimation inside the network, stronger comparison methods (including Parseval regularization), and conflict-guided replay.

2. **`agent.py`** (188 lines):
   - Implements `QNet` (MLP: 4 inputs $\to$ 128 $\to$ 128 $\to$ 2 outputs).
   - Implements basic `ReplayBuffer`.
   - Implements `DQNAgent` with three modes:
     - `naive`: buffer cleared at task boundary, standard DQN updates.
     - `replay`: stores `replay_mem_per_task` (default 3,000) transitions per completed task, mixes old data at fixed ratio `replay_frac=0.5`.
     - `ewc`: calculates diagonal empirical Fisher information from policy distribution induced via softmax over Q-values: $\pi(a|s) = \text{softmax}(Q(s, \cdot))$.
   - **Identified Limitations / Bugs in Existing Code:**
     - Fixed `replay_frac=0.5` regardless of task similarity: when tasks are conflicting, replaying conflicting transitions directly pollutes TD target estimation and destabilizes the current policy; when tasks are identical, it fails to prioritize critical support points.
     - EWC on DQN suffers from value scale shift: Q-values change scale across tasks, distorting the softmax gradient magnitude. Furthermore, quadratic weight penalties freeze shared representation layers, destroying network plasticity.
     - No CLI, no structured YAML configuration, no automated unit test suite, hardcoded hyperparameters.

3. **`continual.py`** (94 lines):
   - Defines sequential loop: `run_sequence(strategy, seq_name, seed, steps_per_task=20000)`.
   - Computes return matrix $R \in \mathbb{R}^{(N+1) \times N}$.
   - Calculates basic metrics: `avg_final_return`, `plasticity`, `forgetting`, `forward_transfer`.
   - **Identified Limitations:**
     - Lacks modular metric definitions adhering to modern CRL literature (backward transfer $BWT$, catastrophic forgetting $F$, forward transfer $FWT$, plasticity preservation).
     - Only evaluates at the end of each task stage, missing fine-grained intra-task learning curve dynamics.
     - Output is unformatted terminal prints; no structured JSON/CSV logging or automated publication plotting pipeline.

4. **`envs.py`** (83 lines):
   - Configures CartPole variants using standard Gymnasium CartPole-v1:
     - `TASK_SEQUENCES['similar']`: physics variations ($g \in [9.8, 12.0]$, $l \in [0.5, 0.9]$, $F \in [9.0, 10.0]$).
     - `TASK_SEQUENCES['partial']`: physics + `FlipWhenRight` wrapper ($x > 0 \implies a \leftarrow 1-a$).
     - `TASK_SEQUENCES['dissimilar']`: physics + `FlipActions` wrapper ($a \leftarrow 1-a$).
   - Good modular basis for task definition; can be formalized into a clean continual benchmark interface with formal similarity metrics.

5. **`fix_results.py`** (6 lines):
   - Scratch script for filtering out partial run keys from legacy JSON logs.

---

## 4. Retained vs. Re-architected Components

| Component | Status | Action Plan |
| :--- | :--- | :--- |
| **Physics Sequences (`envs.py`)** | Retained & Formalized | Keep the three graduated conflict levels (similar, partial, dissimilar); wrap into a clean Gymnasium task generator with deterministic seeding. |
| **Core DQN Agent Structure** | Retained & Refactored | Preserve the clean PyTorch DQN foundation, but refactor into a modular architecture supporting plug-and-play continual learning strategies. |
| **Baselines (Naive, Replay, EWC)** | Upgraded & Fixed | Implement proper baseline hierarchy: Naive Sequential Fine-Tuning, Standard Reservoir Experience Replay, Elastic Weight Consolidation (EWC), and Parseval Spectral Regularization (NeurIPS 2024). |
| **Proposed Contribution** | Brand New | Design and implement **Conflict-Aware Dynamic Plasticity & Replay (CADPR)** or **Task-Similarity Adaptive Replay with Parseval Orthogonality**, addressing both negative interference under conflict and plasticity loss across sequential stages. |
| **Infrastructure & CLI** | Brand New | Build unified `train.py`, `evaluate.py`, `scripts/plot_results.py`, YAML configurations in `configs/`, and comprehensive unit tests in `tests/`. |
