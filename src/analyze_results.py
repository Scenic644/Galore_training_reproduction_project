"""
Collects metrics.json from each run under results/ into one markdown table
(paste straight into README.md) and two plots (rank vs memory, rank vs
accuracy, with AdamW/LoRA as horizontal reference lines).

Written by: Muhammad Umer, Duaa Naz, Reham Hafeez

Usage (after all six configs have finished training):
    python src/analyze_results.py
"""
import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt

def load_metrics():
    # 1. Try reading from results.csv first (the complete master run)
    csv_path = "results/results.csv" if os.path.exists("results/results.csv") else "results.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        rows = []
        for _, row in df.iterrows():
            r_val = None if str(row["rank"]).strip() in ["-", "nan", "None"] else int(float(row["rank"]))
            # Convert GB to MB for consistent plotting
            mem_mb = float(row["peak_vram_gb"]) * 1024.0 if "peak_vram_gb" in row else float(row.get("peak_memory_mb", 0))
            rows.append({
                "method": str(row["method"]).strip(),
                "rank": r_val,
                "peak_memory_mb": mem_mb,
                "accuracy": float(row["eval_accuracy"]) if "eval_accuracy" in row else float(row.get("accuracy", 0))
            })
        return rows

    # 2. Fallback to searching for individual metrics.json files
    rows = []
    for path in sorted(glob.glob("results/*/metrics.json")):
        with open(path) as f:
            rows.append(json.load(f))
    return rows

def print_markdown_table(rows):
    print("\n| Method | Rank | Peak GPU Memory (MB) | Val Accuracy |")
    print("|--------|------|-----------------------|---------------|")
    for r in rows:
        rank_str = str(r["rank"]) if r["rank"] is not None else "—"
        mem_str = f"{r['peak_memory_mb']:.1f}" if r.get("peak_memory_mb") else "n/a"
        acc_str = f"{r['accuracy']:.4f}" if r.get("accuracy") else "n/a"
        print(f"| {r['method']} | {rank_str} | {mem_str} | {acc_str} |")
    print()

def plot_rank_curves(rows, out_path="results/rank_curves.png"):
    galore_rows = sorted(
        [r for r in rows if r["method"] == "galore"],
        key=lambda r: r["rank"]
    )
    if not galore_rows:
        print("No GaLore runs found yet, skipping plot.")
        return

    ranks = [r["rank"] for r in galore_rows]
    mem = [r["peak_memory_mb"] for r in galore_rows]
    acc = [r["accuracy"] for r in galore_rows]

    adamw = next((r for r in rows if r["method"] == "adamw"), None)
    lora = next((r for r in rows if r["method"] == "lora"), None)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=150)

    # Plot 1: Peak Memory
    ax[0].plot(ranks, mem, marker="o", lw=2, color="#e74c3c", label="GaLore")
    if adamw and adamw.get("peak_memory_mb"):
        ax[0].axhline(adamw["peak_memory_mb"], color="grey", linestyle="--", label=f"AdamW ({adamw['peak_memory_mb']:.0f} MB)")
    if lora and lora.get("peak_memory_mb"):
        ax[0].axhline(lora["peak_memory_mb"], color="orange", linestyle="--", label=f"LoRA ({lora['peak_memory_mb']:.0f} MB)")
    ax[0].set_xscale("log", base=2)
    ax[0].set_xticks([4, 16, 64, 128])
    ax[0].set_xticklabels(["4", "16", "64", "128"])
    ax[0].set_xlabel("Rank")
    ax[0].set_ylabel("Peak GPU memory (MB)")
    ax[0].set_title("Rank vs peak memory")
    ax[0].grid(True, linestyle="--", alpha=0.5)
    ax[0].legend()

    # Plot 2: Validation Accuracy
    ax[1].plot(ranks, acc, marker="o", lw=2, color="#2ecc71", label="GaLore")
    if adamw and adamw.get("accuracy"):
        ax[1].axhline(adamw["accuracy"], color="grey", linestyle="--", label=f"AdamW ({adamw['accuracy']:.4f})")
    if lora and lora.get("accuracy"):
        ax[1].axhline(lora["accuracy"], color="orange", linestyle="--", label=f"LoRA ({lora['accuracy']:.4f})")
    ax[1].set_xscale("log", base=2)
    ax[1].set_xticks([4, 16, 64, 128])
    ax[1].set_xticklabels(["4", "16", "64", "128"])
    ax[1].set_xlabel("Rank")
    ax[1].set_ylabel("Validation accuracy")
    ax[1].set_title("Rank vs accuracy")
    ax[1].grid(True, linestyle="--", alpha=0.5)
    ax[1].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    rows = load_metrics()
    if not rows:
        print("No results found in results.csv or results/*/metrics.json.")
    else:
        print_markdown_table(rows)
        plot_rank_curves(rows)
