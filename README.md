# GaLore Reproduction: Memory-Efficient LLM Fine-Tuning

## Overview
Reproduction of "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection" (ICML 2024, Oral).

**Paper:** Zhao et al., "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection," ICML 2024.

## What We Reproduced
We reproduce the paper's RoBERTa-base fine-tuning experiment on SST-2 (GLUE benchmark). The paper reports that GaLore matches full fine-tuning within statistical noise (85.89 vs 85.61 average GLUE) while reducing optimizer memory by up to 65.5%.

## Repository Structure
(insert structure)

## Setup
(insert exact commands)

## Reproduction Results
(insert your table of results)

## Experiment: Hyperparameter Study
**Question:** How does the projection rank used in GaLore affect the trade-off between GPU memory savings and fine-tuning accuracy on SST-2?

**Configurations:** AdamW (baseline), LoRA (r=8), GaLore (r=4, 16, 64, 128).

**Results:** (insert your table + plots)

**Analysis:** (interpret your findings — did lower rank save memory at the cost of accuracy? Where is the sweet spot?)

## Design Choices
- **Dataset:** SST-2 (67k train, 872 val), accessible via HuggingFace Datasets
- **Model:** RoBERTa-base (125M params), following the paper's own fine-tuning setup
- **Optimizer:** HuggingFace Trainer with `optim="galore_adamw"`
- **Memory measurement:** `torch.cuda.max_memory_allocated()` after training
- **Epochs:** 3 (default for GLUE fine-tuning)

## Limitations
- We could not reproduce the paper's LLaMA pre-training experiments (19.7B tokens, multi-GPU A100/H100)
- Our numbers may differ from the paper's due to: different hardware, batch size, random seed, library versions
- SST-2 is a single task; the paper evaluates on 8 GLUE tasks

## Provenance
| Component | Source |
|-----------|--------|
| GaLore projector | Adapted from jiaweizzhao/GaLore |
| GaLoreAdamW | Adapted from jiaweizzhao/GaLore |
| Training pipeline | Written by us |
| Data loading | Written by us (HuggingFace Datasets) |
| Evaluation | Written by us |

## References
- Zhao et al. (2024). GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection. ICML 2024.
- Official repo: https://github.com/jiaweizzhao/GaLore
