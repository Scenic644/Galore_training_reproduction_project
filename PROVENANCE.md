# Provenance

## Written by us
- `src/data.py` — SST-2 loading and tokenization
- `src/train.py` — training orchestration, config handling, memory measurement
- `configs/*.yaml` — all configuration files
- `notebooks/analysis.ipynb` — results analysis and plotting

## Adapted from existing repositories
- `src/galore_projector.py` — adapted from `jiaweizzhao/GaLore/blob/master/galore_torch/galore_projector.py`. Changes: simplified interface, added type hints, documented.
- `src/galore_optimizer.py` — adapted from `jiaweizzhao/GaLore/blob/master/galore_torch/adamw.py`. Changes: wrapped as subclass, added rank parameter handling.

## Reused as-is
- `galore_torch` package (installed via pip from official repo) — used directly in experiments via HuggingFace `optim="galore_adamw"`

## Results reported by original authors
- Paper's GLUE scores: GaLore 85.89 avg, full FT 85.61, LoRA 85.21

## Results obtained by our team
- All numbers in README.md "Reproduction Results" section
