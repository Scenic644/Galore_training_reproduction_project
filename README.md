# GaLore Reproduction: Memory-Efficient LLM Fine-Tuning

Muhammad Umer, Duaa Naz, Reham Hafeez — IBA Karachi

## Overview

This repository reproduces *GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection* (Zhao et al., ICML 2024). Training large models spends most of its memory on the optimizer's bookkeeping rather than on the weights themselves: Adam keeps two extra numbers per parameter, roughly tripling memory use. GaLore reduces this by periodically projecting each weight matrix's gradient into a smaller subspace before the optimizer stores statistics about it, then projecting the update back to full size before applying it. Unlike LoRA, the weights are never frozen or reduced, so the model keeps full-parameter learning capacity.

## What We Reproduced

The paper's headline experiments pretrain LLaMA-style models on up to 19.7 billion tokens of C4 across multiple A100/H100 GPUs, which is out of reach on course-project compute. We instead reproduce the paper's own smaller fine-tuning experiment: RoBERTa-base on SST-2 (GLUE), for which the authors also report results (GaLore 85.89 average GLUE score vs 85.61 for full fine-tuning vs 85.21 for LoRA).

## Repository Structure

```
galore-reproduction/
├── README.md
├── requirements.txt
├── src/
│   ├── data.py              # SST-2 loading and tokenization
│   ├── train.py             # training for AdamW / LoRA / GaLore
│   └── analyze_results.py   # collects metrics.json into a table + plots
├── configs/
│   ├── adamw_baseline.yaml
│   ├── lora_r8.yaml
│   ├── galore_r4.yaml
│   ├── galore_r16.yaml
│   ├── galore_r64.yaml
│   └── galore_r128.yaml
└── results/                 # created by train.py, one folder per run
```

## Setup

```bash
pip install -r requirements.txt
```

On Colab, run each config from its own cell. Do the sanity check first:

```bash
python src/train.py configs/adamw_baseline.yaml --sanity
python src/train.py configs/adamw_baseline.yaml
```

Repeat for the other five configs. Each RoBERTa-base/SST-2 run takes roughly 20 to 40 minutes on a free-tier T4.

Once all six runs have finished:

```bash
python src/analyze_results.py
```

This prints the results table below and saves `results/rank_curves.png`.

## Method

<!-- 3-5 sentences, in your own words, on Algorithm 1: how P and Q are found via SVD,
how often they're refreshed (update_proj_gap), and how the projected update is scaled
back (scale). Pull this from your Stage 2 notes rather than rephrasing the abstract. -->

## Reproduction Results

<!-- Paste the table printed by `python src/analyze_results.py` here. -->

| Method | Rank | Peak GPU Memory (MB) | Val Accuracy |
|--------|------|-----------------------|---------------|
| AdamW  | —    |                       |               |
| LoRA   | 8    |                       |               |
| GaLore | 4    |                       |               |
| GaLore | 16   |                       |               |
| GaLore | 64   |                       |               |
| GaLore | 128  |                       |               |

![Rank vs memory and accuracy](results/rank_curves.png)

## Experiment: Hyperparameter Study

**Question:** How does the projection rank used in GaLore affect the trade-off between GPU memory savings and fine-tuning accuracy on SST-2?

**Configurations:** AdamW (full fine-tuning), LoRA (rank 8), GaLore (ranks 4, 16, 64, 128).

**Analysis:** <!-- Once you have numbers: where does lower rank start costing accuracy?
Where's the sweet spot? Does GaLore's memory saving actually show up on a model this
small, or only at larger scale? Say so either way, that's a legitimate finding. -->

## Design Choices

- Dataset: SST-2 (67k train, 872 validation), via HuggingFace Datasets
- Model: RoBERTa-base (125M parameters)
- GaLore applied to `query`, `key`, `value`, and `dense`, RoBERTa's actual linear layer names. GaLore's own examples default to `attn`/`mlp`, which are LLaMA-style names and match nothing in RoBERTa
- Optimizer: HuggingFace `Trainer` with `optim="galore_adamw"`
- Memory measurement: `torch.cuda.max_memory_allocated()` after each run
- Epochs: 3 (standard for GLUE fine-tuning)

## Limitations

- We could not reproduce the paper's LLaMA/C4 pretraining experiments; compute constraints
- SST-2 is a single GLUE task, not the paper's full 8-task average
<!-- Add anything else once you see your numbers: batch size, seed, or hyperparameter
differences from the paper, and why your numbers might diverge from theirs. -->

## Provenance

| Component | Source |
|-----------|--------|
| `src/data.py`, `src/train.py`, `src/analyze_results.py`, all configs | Written by us |
| GaLore optimizer (`galore_adamw`) | Reused as-is, via `galore-torch` / HuggingFace `Trainer` integration |
| LoRA (`peft.LoraConfig`) | Reused as-is, via HuggingFace `peft` |
| Paper's GLUE scores (85.89 / 85.61 / 85.21) | Reported by the original authors |
| All numbers in the Reproduction Results table above | Obtained by our team |

## References

- Zhao et al. (2024). GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection. ICML 2024.
- Official repo: https://github.com/jiaweizzhao/GaLore
