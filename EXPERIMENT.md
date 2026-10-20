# Experiments (branch `milestone-3-experiments`)

Two additional experiments on top of the Milestone 2 reproduction:

1. Seed robustness: is GaLore's accuracy gap to AdamW larger than seed-to-seed noise?
2. GaLore's scale parameter: does using the paper's value of α close our gap to the paper at low rank?

Both use RoBERTa-base fine-tuned on GLUE SST-2 (3 epochs, batch size 16, max length 128, fp32, NVIDIA T4), evaluated on the 872-example validation set after the last epoch. Unless stated, the learning rate is 2e-5 and GaLore uses the library default scale α = 0.25.

---

## Experiment 1: Seed robustness (AdamW vs GaLore rank 128)

### Research question
Is GaLore's accuracy gap to AdamW at rank 128 larger than seed-to-seed noise?

### Setup
Each method was trained with four random seeds (1, 2, 3 and 42). Runs were split across several Colab sessions, with identical code and settings; only `--seed` differed.

```bash
# AdamW baseline
python src/train.py --method adamw --seed N

# GaLore, rank 128
python src/train.py --method galore --rank 128 --seed N
```

The seed 42 runs also used `--save_model`, and those weights are the models served by the demo app (`app/app.py`). Every run is logged in `results/seed_runs.csv`; `python src/analyze_seeds.py` (run from the repo root) reproduces the summary below.

### Results

| Seed | AdamW accuracy (%) | GaLore r=128 accuracy (%) | Gap (AdamW minus GaLore) |
| :---: | :---: | :---: | :---: |
| 1 | 93.69 | 93.69 | 0.00 |
| 2 | 93.58 | 93.23 | 0.35 |
| 3 | 93.81 | 93.23 | 0.58 |
| 42 | 94.15 | 93.92 | 0.23 |
| **Mean (SD)** | **93.81 (0.25)** | **93.52 (0.34)** | **0.29 (0.24)** |

| | AdamW | GaLore r=128 |
| :--- | :---: | :---: |
| Mean peak VRAM (GB) | 2.063 | 1.840 (10.8% lower) |
| Mean training time (s) | 1531 | 2580 (1.7x longer) |

### Conclusion
GaLore at rank 128 used about 11% less peak GPU memory than AdamW and took about 1.7 times as long to train, because of the periodic SVD of the gradients. Its accuracy was lower than or equal to AdamW's on all four seeds, by 0.29 points on average.

That gap is small. It is smaller than the roughly 0.8 point sampling error of a single accuracy on 872 validation examples, and about the same size as the gap the paper reports on SST-2. A paired comparison across the four seeds gives a mean gap of 0.29 points with a standard error of 0.12 (t = 2.4 on 3 degrees of freedom, p of about 0.10). The evidence therefore suggests a small, consistent accuracy cost for GaLore at this rank, but four seeds are too few to rule out that it is noise.

This result was measured at α = 0.25. Experiment 2 shows α matters a lot at low rank, so the 0.29-point gap should not be assumed to hold at the paper's α.

### Limitations
- Four seeds and one validation set of 872 examples limit the statistical power.
- Only rank 128 was tested across seeds; the rank sweep in the main README uses seed 42 only.
- Peak VRAM is total memory for a 125M-parameter model, so activations and weights dominate; the optimiser-state saving the paper reports is a smaller share of the total here.

---

## Experiment 2: GaLore with the paper's scale α

### Research question
The Milestone 2 runs and Experiment 1 used the library default α = 0.25 (the paper's pre-training value). The paper's RoBERTa SST-2 recipe (Appendix D.1, Table 7) uses α = 4 for rank 4 and α = 2 for rank 8. GaLore multiplies its update by α, so α = 0.25 gives a step 16 times smaller than α = 4 at the same learning rate. Does using the paper's α close the gap to AdamW at low rank?

### Setup
Three single-run experiments, seed 42, 3 epochs, everything else as above. They add two options to `src/train.py`: `--scale` (GaLore α) and `--results_path`. They do not use `--save_model`, so the demo models are unchanged. Results are in `results/paper_recipe_runs.csv`.

```bash
# Run A: the paper's rank 4 setting (lr 1e-5, alpha 4)
python src/train.py --method galore --rank 4 --lr 1e-5 --scale 4 --seed 42 --results_path results/paper_recipe_runs.csv

# Run B: the paper's rank 8 setting (lr 2e-5, alpha 2)
python src/train.py --method galore --rank 8 --lr 2e-5 --scale 2 --seed 42 --results_path results/paper_recipe_runs.csv

# Run C: rank 4 with our shared lr (differs from the earlier rank 4 run only in alpha)
python src/train.py --method galore --rank 4 --lr 2e-5 --scale 4 --seed 42 --results_path results/paper_recipe_runs.csv
```

### Results

| Run | Rank | lr | α | Accuracy (%) | Peak VRAM (GB) | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Earlier rank 4 run (main README) | 4 | 2e-5 | 0.25 | 92.66 | 1.724 | 2363 |
| A | 4 | 1e-5 | 4 | 93.58 | 1.724 | 2489 |
| C | 4 | 2e-5 | 4 | 94.15 | 1.724 | 2434 |
| B | 8 | 2e-5 | 2 | 94.50 | 1.727 | 2487 |
| AdamW (main README) | n/a | 2e-5 | n/a | 94.15 | 2.062 | 1478 |

For reference, the paper reports 94.04 (rank 4) and 94.38 (rank 8) for GaLore and 94.57 for full fine-tuning, after 30 epochs.

### Conclusion
- Changing only α from 0.25 to 4 (Run C against the earlier rank 4 run: same seed, rank and learning rate) raised accuracy from 92.66% to 94.15%, which is 13 more of the 872 sentences correct. Peak memory did not change (1.724 GB).
- Run C matches AdamW on seed 42 exactly (both 94.15%) with 16.4% less peak memory. Run B is 0.35 points above AdamW (3 sentences), which is within noise; we do not claim GaLore beats AdamW.
- Run A, the paper's own rank 4 setting, scored 93.58%, 5 sentences below Run C. That is within noise. With only 3 of the paper's 30 epochs, a learning rate of 1e-5 is probably undertrained, but we did not test this.
- Our first explanation for the gap to the paper was the shorter schedule. These runs point to the small α instead: with the paper's α, low-rank GaLore lands within about 0.15 points of the paper's GaLore numbers at rank 4 (Run C) and rank 8 (Run B), after 3 epochs.

### Limitations
- Each run is a single seed. One sentence is worth 0.11 points and the sampling error of a single accuracy is about 0.8 points, so only the 1.49-point change from α is clearly larger than the noise we measured in Experiment 1 (seed SD of about 0.3). It is one comparison, so treat it as strong evidence, not a proof.
- Experiment 1 (rank 128) and the rank sweep in the main README were measured at α = 0.25. We did not repeat them at the paper's α, so the "accuracy rises with rank" trend there may partly reflect α, and the 0.29-point gap in Experiment 1 may not hold at α = 4.
- The demo app still serves the α = 0.25 rank 128 model.
