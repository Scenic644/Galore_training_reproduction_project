# Experiment: Seed Robustness of GaLore vs AdamW

## Research question
Is GaLore's accuracy gap to AdamW at rank 128 larger than seed-to-seed noise?

## Setup
RoBERTa-base fine-tuned on GLUE SST-2 (3 epochs, batch size 16, learning rate 2e-5, max length 128, fp32, NVIDIA T4). Each method was trained with four random seeds (1, 2, 3 and 42). Evaluation is accuracy on the 872-example SST-2 validation set. Runs were split across several Colab sessions, with identical code and settings; only `--seed` differed.

```bash
# AdamW baseline
python src/train.py --method adamw --seed N

# GaLore, rank 128
python src/train.py --method galore --rank 128 --seed N
```

The seed 42 runs also used `--save_model`, and those weights are the models served by the demo app (`app/app.py`). Every run is logged in `results/seed_runs.csv`; `python src/analyze_seeds.py` (run from the repo root) reproduces the summary below.

## Results

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

## Conclusion
GaLore at rank 128 used about 11% less peak GPU memory than AdamW and took about 1.7 times as long to train, because of the periodic SVD of the gradients. Its accuracy was lower than or equal to AdamW's on all four seeds, by 0.29 points on average.

That gap is small. It is smaller than the roughly 0.8 point sampling error of a single accuracy on 872 validation examples, and about the same size as the gap the paper reports on SST-2. A paired comparison across the four seeds gives a mean gap of 0.29 points with a standard error of 0.12 (t = 2.4 on 3 degrees of freedom, p of about 0.10). The evidence therefore suggests a small, consistent accuracy cost for GaLore at this rank, but four seeds are too few to rule out that it is noise.

## Limitations
- Four seeds and one validation set of 872 examples limit the statistical power.
- Only rank 128 was tested across seeds; the rank sweep in the main README uses seed 42 only.
- Peak VRAM is total memory for a 125M-parameter model, so activations and weights dominate; the optimizer-state saving the paper reports is a smaller share of the total here.
