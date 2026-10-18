"""Summarise results/seed_runs.csv: mean/std per method, paired gap, noise comparison."""
import math
import pandas as pd

df = pd.read_csv("results/seed_runs.csv").drop_duplicates(subset=["method", "seed"], keep="first")
n_val = 872
piv = df.pivot(index="seed", columns="method", values="eval_accuracy") * 100
gap = piv["adamw"] - piv["galore"]  # positive = AdamW ahead

print(piv.round(2).to_string())
for m in ("adamw", "galore"):
    print(f"{m:7s} mean {piv[m].mean():.2f}  sd {piv[m].std(ddof=1):.2f}  range {piv[m].min():.2f}-{piv[m].max():.2f}")
mem = df.groupby("method")["peak_vram_gb"].mean()
rt = df.groupby("method")["train_runtime_sec"].mean()
print(f"gap (AdamW - GaLore): mean {gap.mean():.2f}, sd {gap.std(ddof=1):.2f}, per seed {gap.round(2).tolist()}")
se = gap.std(ddof=1) / math.sqrt(len(gap))
print(f"paired mean gap {gap.mean():.2f} +/- {se:.2f} (SE over {len(gap)} seeds); t = {gap.mean()/se:.2f}")
p = piv["adamw"].mean() / 100
print(f"binomial SE of one accuracy on {n_val} examples: {100*math.sqrt(p*(1-p)/n_val):.2f} points")
print(f"mean peak VRAM: AdamW {mem['adamw']:.3f} GB, GaLore {mem['galore']:.3f} GB ({100*(1-mem['galore']/mem['adamw']):.1f}% lower)")
print(f"mean runtime: AdamW {rt['adamw']:.0f}s, GaLore {rt['galore']:.0f}s ({rt['galore']/rt['adamw']:.2f}x)")
