"""
Collects metrics.json from each run under results/ into one markdown table
(paste straight into README.md) and two plots (rank vs memory, rank vs
accuracy, with AdamW/LoRA as horizontal reference lines).

Written by: Muhammad Umer, Duaa Naz, Reham Hafeez

Usage (after all six configs have finished training):
    python src/analyze_results.py
"""
import glob
import json

import matplotlib.pyplot as plt


def load_all_metrics(results_dir="results"):
    rows = []
    for path in sorted(glob.glob(f"{results_dir}/*/metrics.json")):
        with open(path) as f:
            rows.append(json.load(f))
    return rows


def print_markdown_table(rows):
    print("\n| Method | Rank | Peak GPU Memory (MB) | Val Accuracy |")
    print("|--------|------|----------------------|---------------|")
    for r in rows:
        rank = r["rank"] if r["rank"] is not None else "\u2014"
        mem = f"{r['peak_memory_mb']:.1f}" if r["peak_memory_mb"] else "n/a"
        print(f"| {r['method']} | {rank} | {mem} | {r['accuracy']:.4f} |")
    print()


def plot_rank_curves(rows, out_path="results/rank_curves.png"):
    galore_rows = sorted(
        (r for r in rows if r["method"] == "galore"), key=lambda r: r["rank"]
    )
    if not galore_rows:
        print("No GaLore runs found yet, skipping plot.")
        return

    ranks = [r["rank"] for r in galore_rows]
    mem = [r["peak_memory_mb"] for r in galore_rows]
    acc = [r["accuracy"] for r in galore_rows]

    adamw = next((r for r in rows if r["method"] == "adamw"), None)
    lora = next((r for r in rows if r["method"] == "lora"), None)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    ax[0].plot(ranks, mem, marker="o", label="GaLore")
    if adamw and adamw["peak_memory_mb"]:
        ax[0].axhline(adamw["peak_memory_mb"], color="grey", linestyle="--", label="AdamW")
    if lora and lora["peak_memory_mb"]:
        ax[0].axhline(lora["peak_memory_mb"], color="orange", linestyle="--", label="LoRA")
    ax[0].set_xlabel("Rank")
    ax[0].set_ylabel("Peak GPU memory (MB)")
    ax[0].set_title("Rank vs peak memory")
    ax[0].legend()

    ax[1].plot(ranks, acc, marker="o", label="GaLore")
    if adamw:
        ax[1].axhline(adamw["accuracy"], color="grey", linestyle="--", label="AdamW")
    if lora:
        ax[1].axhline(lora["accuracy"], color="orange", linestyle="--", label="LoRA")
    ax[1].set_xlabel("Rank")
    ax[1].set_ylabel("Validation accuracy")
    ax[1].set_title("Rank vs accuracy")
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    rows = load_all_metrics()
    if not rows:
        print("No metrics.json files found under results/. Run the six configs first.")
    else:
        print_markdown_table(rows)
        plot_rank_curves(rows)
