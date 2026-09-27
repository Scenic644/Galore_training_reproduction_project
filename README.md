# GaLore Reproduction: Memory-Efficient LLM Fine-Tuning

**Authors:** Muhammad Umer (30622), Duaa Naz (30668), Reham Hafeez (30574)  
**Institution:** Institute of Business Administration (IBA), Karachi  
**Course:** Machine Learning — Milestone 2 Reproduction Study  
**Hardware:** NVIDIA Tesla T4 GPU (16 GB GDDR6 VRAM, Deterministic Seed: 42)

---

## Overview

This repository provides an empirical reproduction and analysis of *GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection* (Zhao et al., ICML 2024). In deep learning optimization, the primary memory bottleneck is not parameter storage, but the stateful regularizers maintained by adaptive optimizers. In standard 16-bit mixed-precision AdamW, maintaining first- ($M_t$) and second-moment ($V_t$) statistics requires 8 bytes per parameter—double the memory footprint of the weights themselves.

GaLore mitigates this bottleneck by exploiting the empirical and theoretical property that the **gradient matrix $G \in \mathbb{R}^{m \times n}$ becomes naturally low-rank during training**. By projecting $G$ into a time-varying compact subspace via periodic Singular Value Decomposition (SVD), optimizer states are tracked strictly in low-rank form. Unlike Low-Rank Adaptation (LoRA), GaLore updates the primary uncompressed weights directly, preserving full-parameter learning dynamics without freezing base matrices.

---

## What We Reproduced

The headline experiment in Zhao et al. evaluates pre-training LLaMA-7B on the C4 corpus across a cluster of 64 A100 GPUs, which is computationally intractable on student compute quotas. In alignment with Stage 3 reproduction guidelines, we reproduced **Section 5.4 / Table 4 of the paper**: memory-efficient downstream fine-tuning of RoBERTa-Base on the GLUE Stanford Sentiment Treebank (SST-2) benchmark. 

This experiment directly evaluates:
1. **Representational Capacity:** Does GaLore maintain full-parameter expressivity over LoRA at low ranks?
2. **Optimizer Memory Scaling:** Does GaLore reduce peak memory footprint without destabilizing convergence?
3. **Throughput & Latency Trade-offs:** What is the actual computational overhead introduced by periodic SVD subspace factorizations?

---

## Repository Structure

```
galore-reproduction/
├── README.md                        # Complete reproduction report, analysis, and paper QA
├── PROVENANCE.md                    # Explicit component attribution and code classification
├── requirements.txt                 # Pinned environment dependencies
├── galore_reproduction_colab.ipynb # End-to-end Google Colab execution runner
├── Milestone_2_report.pdf
├── configs/                         # Initial configuration drafts for YAML runner
│   ├── adamw_baseline.yaml
│   ├── lora_r8.yaml
│   ├── galore_r4.yaml
│   ├── galore_r16.yaml
│   ├── galore_r64.yaml
│   └── galore_r128.yaml
├── src/
│   ├── train.py                     # Primary training orchestrator (CLI: --method and --rank)
│   ├── sanity_check.py              # Standalone verification script (forward/backward & shape check)
│   ├── analyze_results.py           # Aggregates metrics from results.csv and plots Pareto curves
│   ├── data.py                      # SST-2 tokenization and dataset preprocessing
│   └── train_yaml.py                # Legacy YAML-based training script (initial draft)
└── results/
    ├── .gitkeep
    └── results.csv                  # Verified 6-configuration empirical metrics
```

---

## Setup & Execution

### 1. Environment Installation
Install the pinned dependencies:
```bash
pip install -r requirements.txt
```

### 2. Sanity Verification Pass
Before launching full training runs, execute the standalone sanity check:
```bash
python src/sanity_check.py
```
*Expected behavior:* Loads a 16-sample slice of SST-2 (`nyu-mll/glue`), executes one forward and backward pass through RoBERTa-Base, verifies output logits shape `(8, 2)` and initial finite loss `0.6696`, and outputs:
```text
Sanity check passed. Safe to launch a full training run.
```

### 3. Running Experimental Configurations
Execute each configuration from the repository root. Each run trains for 3 epochs (12,630 steps, batch size 16) and appends its validation accuracy, peak VRAM, and runtime directly to `results/results.csv`:

```bash
# 1. Full-rank AdamW Baseline
python src/train.py --method adamw

# 2. LoRA (Rank 8 PEFT Baseline)
python src/train.py --method lora --rank 8

# 3. GaLore Hyperparameter Rank Sweep (Ranks 4, 16, 64, 128)
python src/train.py --method galore --rank 4
python src/train.py --method galore --rank 16
python src/train.py --method galore --rank 64
python src/train.py --method galore --rank 128
```
*(Benchmark runtimes on NVIDIA Tesla T4: LoRA ~16 min, AdamW ~25 min, GaLore ~40 min per configuration).*

### 4. Compiling Results & Generating Pareto Plot
To print the summary markdown table and generate the rank curves plot (`results/rank_curves.png`):
```bash
python src/analyze_results.py
```

---

## Method and Algorithmic Formulation

GaLore optimizes linear layers $W \in \mathbb{R}^{m \times n}$ by projecting weight gradients into low-rank subspaces rather than parameterizing weights into low-rank matrices. Every $T$ steps (here, $T=200$), GaLore performs a truncated Singular Value Decomposition (SVD) on the layer gradient $G_t = -\nabla_W \mathcal{L}(W_t) \in \mathbb{R}^{m \times n}$:
$$G_t = U \Sigma V^T \implies P_t = U[:, :r] \in \mathbb{R}^{m \times r}$$
where $P_t$ satisfies $P_t^T P_t = I_r$ (assuming $m \le n$). 

The gradient is projected into compact representation:
$$R_t = P_t^T G_t \in \mathbb{R}^{r \times n}$$
Adam updates first ($M_t$) and second ($V_t$) moments directly within $\mathbb{R}^{r \times n}$. The normalized step $N_t = \frac{M_t}{\sqrt{V_t} + \epsilon}$ is mapped back to the original parameter space:
$$\tilde{G}_t = \alpha (P_t N_t) \in \mathbb{R}^{m \times n}$$
$$W_t = W_{t-1} + \eta \tilde{G}_t$$

Crucially, scale factor $\alpha$ is constant and independent of rank $r$ (unlike LoRA's $\frac{\alpha}{r}$), preventing gradient vanishing at small ranks. Model parameters $W$ remain unconstrained and full-rank throughout optimization.

---

## Reproduction Results

The complete empirical matrix across all 6 verified configurations on GLUE SST-2 (NVIDIA Tesla T4 GPU, batch size 16, deterministic `seed=42`):

| Optimization Method | Rank ($r$) | Peak VRAM (MB) | Peak VRAM (GB) | Memory Saved vs. AdamW | Val Accuracy (%) | Runtime (s) | Throughput (it/s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full AdamW (Baseline)** | — | 2111.5 MB | 2.062 GB | 0.0% | **94.15%** | 1477.8 s | 8.55 it/s |
| **LoRA** | 8 | **1315.5 MB** | **1.285 GB** | **37.7%** | 91.28% | **982.7 s** | **12.85 it/s** |
| **GaLore** | 4 | 1765.4 MB | 1.724 GB | **16.4%** | 92.66% | 2363.1 s | 5.34 it/s |
| **GaLore** | 16 | 1775.6 MB | 1.734 GB | 15.9% | 92.89% | 2369.7 s | 5.33 it/s |
| **GaLore** | 64 | 1816.6 MB | 1.774 GB | 14.0% | 93.58% | 2408.8 s | 5.24 it/s |
| **GaLore** | 128 | 1885.2 MB | 1.841 GB | **10.7%** | **93.92%** | 2494.5 s | 5.06 it/s |

---

## Experiment: Hyperparameter Study (Rank Sweep Analysis)

**Research Question:** How does the projection rank $r \in \{4, 16, 64, 128\}$ affect the trade-off between physical GPU memory savings, execution latency, and downstream accuracy on SST-2?

### Key Findings ("What We Found"):
1. **GaLore Outperforms LoRA in Representational Expressivity:**
   At rank $r=8$, LoRA achieved 91.28% accuracy. GaLore at half the rank capacity ($r=4$) attained 92.66% (+1.38% over LoRA). At $r=128$, GaLore achieves 93.92%, coming within **0.23%** of unconstrained full-rank AdamW (94.15%). This confirms that optimizing within dynamic, time-varying low-rank gradient subspaces avoids the capacity ceiling of frozen weights inherent to LoRA.
2. **Optimizer Memory Scaling vs. Rank:**
   GaLore reduces peak training VRAM from 2.062 GB (AdamW) down to 1.724 GB (16.4% savings at $r=4$). Scaling the rank from $r=4$ to $r=128$ increases peak VRAM by only **117 MB** (1.724 GB → 1.841 GB, 10.7% savings), demonstrating that GaLore suppresses optimizer state expansion effectively across ranks.
3. **The Compute/Throughput Trade-off:**
   GaLore introduced an approximate **60% runtime overhead** (~2400 s vs. 1477.8 s for AdamW). This empirically confirms the computational cost of periodic SVD factorizations ($T=200$) and bidirectional matrix projections ($P^T G$ and $P N_t$), representing a direct trade-off between device memory conservation and wall-clock execution speed.

---

## Stage 2: Paper Understanding (The 8 Core Questions)

### Q1: What is the problem, and why does it matter?
* **Problem:** Large language model training is bounded by GPU memory. The primary bottleneck is not model parameters, but **optimizer states**. In standard 16-bit AdamW, first and second momentum states ($M_t, V_t$) consume 8 bytes per parameter—double the memory of the weights themselves.
* **Why it matters:** Full pre-training and fine-tuning of 7B+ parameter models historically requires multi-GPU server clusters (e.g., $8\times\text{A100}$), placing them out of reach for consumer GPUs with $\le 24\text{ GB}$ VRAM.

### Q2: What did people do before this paper, and what was missing?
* **Prior Work:** LoRA freezes base weights $W_0$ and adds trainable low-rank adapters ($W = W_0 + BA$). ReLoRA attempts pre-training by periodically merging adapters into base weights.
* **What was missing:** LoRA restricts parameter search to a static, time-invariant low-rank subspace throughout training, altering gradient dynamics and degrading pre-training unless preceded by an expensive full-rank warmup.

### Q3: What is the proposed solution, concretely?
* **GaLore (Gradient Low-Rank Projection):** Instead of assuming weights are low-rank, GaLore leverages the theoretical finding that the **gradient matrix $G \in \mathbb{R}^{m \times n}$ becomes naturally low-rank during training**. It projects $G$ into a compact subspace ($P^T G$), maintains Adam optimizer states in that compact space, and projects updates back to update full-rank weights directly.

### Q4: What's actually novel?
* **Full-Parameter Optimization:** Updates full-rank weights directly ($W \leftarrow W - \eta \tilde{G}$), preserving full parameter search.
* **Low-Rank Gradient vs. Low-Rank Weights:** Applies compression to the gradient, not the model weights.
* **Dynamic Subspace Trajectories:** Recomputes projection operators via SVD every $T$ steps (e.g., $T=200$), allowing the optimizer to traverse multiple low-rank subspaces over time.
* **Rank-Independent Scale $\alpha$:** Unlike LoRA's $\frac{\alpha}{r}$ scaling, GaLore sets $\alpha$ constant, preventing gradient vanishing at low ranks.

### Q5: What's the full algorithm, step by step?
1. **Gradient Computation:** Evaluate backward pass $G_t \leftarrow -\nabla_W \mathcal{L}(W_t)$.
2. **Subspace Re-initialization:** Every $T$ steps, perform SVD: $G_t = U \Sigma V^T \implies P_t = U[:, :r]$. Else, reuse $P_t = P_{t-1}$.
3. **Low-Rank Projection:** Project gradient into compact space: $R_t = P_t^T G_t \in \mathbb{R}^{r \times n}$.
4. **Low-Rank Optimizer Update:** Update Adam momentum states in $\mathbb{R}^{r \times n}$:
   $$M_t = \beta_1 M_{t-1} + (1-\beta_1) R_t, \quad V_t = \beta_2 V_{t-1} + (1-\beta_2) R_t^2$$
5. **Normalize:** $N_t = \frac{M_t / (1-\beta_1^t)}{\sqrt{V_t / (1-\beta_2^t)} + \epsilon}$.
6. **Project Back & Update:** $\tilde{G}_t = \alpha (P_t N_t)$, and update full weights: $W_t = W_{t-1} + \eta \tilde{G}_t$.

### Q6: What dataset(s), and why those?
* The paper evaluated pre-training on **C4** and downstream fine-tuning on **GLUE** (RoBERTa-Base).
* **Our Reproduction:** Due to academic single-GPU compute quotas (NVIDIA Tesla T4 with 16 GB VRAM vs. 64$\times$A100 GPUs), we replicated **Section 5.4 / Table 4 (GLUE SST-2 fine-tuning)**. SST-2 has 67,349 training and 872 validation sequences, isolating memory scaling and representational expressivity under controlled conditions.

### Q7: What metrics, and what do they actually measure (and fail to measure)?
* **Evaluation Accuracy:** Measures classification correctness on SST-2 validation split. *Fails to measure:* Model calibration, confidence error, or out-of-distribution robustness.
* **Peak VRAM Allocation (GB):** Measures maximum hardware memory allocated via `torch.cuda.max_memory_allocated()`. *Fails to measure:* Per-layer transient activation spikes or cache fragmentation.
* **Training Runtime (s) & Throughput (it/s):** Measures wall-clock execution time. *Fails to measure:* Pure SVD projection kernel latency isolated from PyTorch overhead.

### Q8: What were the headline results, and what did the authors flag as limitations?
* **Headline Results:** Feasibility of pre-training LLaMA-7B on a single 24 GB GPU (RTX 4090); up to 65.5% optimizer memory reduction.
* **Author-Flagged Limitations:** (1) SVD computational latency overhead (~17%); (2) Mathematical proof relies on reversible network assumptions, requiring empirical extensions (JoMA) for Transformers; (3) Subspace frequency $T$ requires manual tuning.

---

## Design Choices & Implementation Details

- **Dataset:** SST-2 (67k train, 872 validation), loaded via Hugging Face `datasets`.
- **Model:** RoBERTa-base (125M parameters).
- **Target Modules:** In RoBERTa, GaLore was applied to `query`, `key`, `value`, and `dense` layers (matching RoBERTa's actual module names, whereas GaLore's pretraining defaults target LLaMA's `q_proj`, `k_proj`, etc.).
- **Optimizer Integration:** Integrated via `galore-torch==1.0` and Hugging Face `Trainer` with `optim="galore_adamw"`.
- **Memory Instrumentation:** Monitored using PyTorch hardware hooks via `torch.cuda.max_memory_allocated()` reset prior to every run.
- **Reproducibility:** Seed fixed to `42` across all runs to ensure deterministic evaluations.

---

## Environment Hygiene & Bug Resolutions

During initial setup on Google Colab, an upstream `HfUriError` occurred when querying `load_dataset("glue", "sst2")`:
* **Cause:** `huggingface_hub >= 0.25.0` enforced strict repository namespace validation (`namespace/name`), rejecting the legacy un-namespaced string `"glue"`.
* **Fix:** Patched all dataset loading scripts in `src/data.py`, `src/train.py`, and `src/sanity_check.py` to use the canonical namespaced path:
  ```python
  "glue" -> "nyu-mll/glue"
  ```
  Verified via `src/sanity_check.py` passing with forward logits shape `(8, 2)` and initial finite loss `0.6696`.

---

## Limitations

- **Compute Boundaries:** We could not reproduce the 7B LLaMA pre-training runs on C4 due to cluster-scale compute requirements (64$\times$A100 GPUs).
- **Task Scope:** We evaluated on GLUE SST-2 (binary sentiment) rather than the complete 8-task GLUE benchmark suite.
- **SVD Wall-Clock Overhead:** While GaLore achieved significant memory reduction, SVD subspace recalculation every 200 steps resulted in a ~60% wall-clock slowdown on Tesla T4 hardware compared to full AdamW.

---

## Code Provenance

| Component | Source / Classification |
|---|---|
| `src/train.py`, `src/sanity_check.py`, `src/analyze_results.py`, `src/data.py` | **Written / Adapted by our team** |
| GaLore optimizer (`galore_adamw`) | **Reused as-is**, via `galore-torch==1.0` / Hugging Face `Trainer` integration |
| LoRA (`peft.LoraConfig`) | **Reused as-is**, via Hugging Face `peft` |
| Paper baseline GLUE scores | **Reported by original authors** (Zhao et al., ICML 2024) |
| Empirical numbers in results table | **Obtained experimentally by our team on NVIDIA Tesla T4** |

---

## References

1. Zhao et al. (2024). *GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection*. ICML 2024. [arXiv:2403.03507](https://arxiv.org/abs/2403.03507)
2. Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
3. Loshchilov & Hutter (2019). *Decoupled Weight Decay Regularization*. ICLR 2019. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
4. Lialin et al. (2024). *ReLoRA: High-Rank Training Through Low-Rank Updates*. ICLR 2024. [arXiv:2307.05695](https://arxiv.org/abs/2307.05695)
5. Liu et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach*. [arXiv:1907.11692](https://arxiv.org/abs/1907.11692)
6. Wang et al. (2019). *GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding*. ICLR 2019. [arXiv:1804.07461](https://arxiv.org/abs/1804.07461)
