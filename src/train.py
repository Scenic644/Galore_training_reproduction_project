"""
train.py -- Fine-tune RoBERTa-base on SST-2 under one of six configurations:

    python train.py --method adamw
    python train.py --method lora   --rank 8
    python train.py --method galore --rank 4
    python train.py --method galore --rank 16
    python train.py --method galore --rank 64
    python train.py --method galore --rank 128

Run sanity_check.py once before this to confirm the model, data, and loss
all behave as expected on a tiny batch.

Extra options on this branch: --seed, --scale (GaLore alpha, default 0.25),
--lr, --save_model (writes models/<run_name> for the demo app) and
--results_path.

Each call appends one row to results/seed_runs.csv with:
    method, rank, seed, eval_accuracy, peak_vram_gb, train_runtime_sec
If --results_path points elsewhere, the row also records lr and scale. Do not
point it at an existing file with a different header, such as the Milestone 2
results/results.csv, which has no seed column.
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

MODEL_NAME = "roberta-base"
RESULTS_PATH = "results/seed_runs.csv"  # experiment branch: kept apart from main's results.csv
DEFAULT_SEED = 42

# Regex targets for GaLore: every attention and feed-forward weight matrix
# inside RoBERTa's encoder layers. Deliberately excludes the classification
# head and embeddings -- they're small, and not where the paper's memory
# claim is being tested.
GALORE_TARGET_MODULES = [
    r"^roberta\.encoder\.layer\.\d+\.attention\.self\.(query|key|value)$",
    r"^roberta\.encoder\.layer\.\d+\.attention\.output\.dense$",
    r"^roberta\.encoder\.layer\.\d+\.intermediate\.dense$",
    r"^roberta\.encoder\.layer\.\d+\.output\.dense$",
]


def parse_args():
    parser = argparse.ArgumentParser(description="GaLore reproduction: RoBERTa-base on SST-2")
    parser.add_argument("--method", choices=["adamw", "lora", "galore"], required=True)
    parser.add_argument(
        "--rank", type=int, default=None,
        help="Required for lora (use 8) and galore (use 4, 16, 64, or 128)",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="./runs")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--scale", type=float, default=0.25,
        help="GaLore scale alpha (default 0.25; the paper's fine-tuning recipe uses 4 at rank 4 and 2 at rank 8)",
    )
    parser.add_argument(
        "--results_path", type=str, default=RESULTS_PATH,
        help="CSV to append to; a non-default path also records lr and scale",
    )
    parser.add_argument(
        "--save_model", action="store_true",
        help="Save the final weights and tokenizer to models/<run_name> for the demo app",
    )
    return parser.parse_args()


def load_data(tokenizer, max_length):
    raw = load_dataset("nyu-mll/glue", "sst2")

    def tokenize_fn(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=max_length)

    tokenized = raw.map(tokenize_fn, batched=True, remove_columns=["sentence", "idx"])
    tokenized = tokenized.rename_column("label", "labels")
    return tokenized["train"], tokenized["validation"]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float((preds == labels).mean())}


def build_model(method, rank):
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    if method == "lora":
        from peft import LoraConfig, TaskType, get_peft_model

        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=rank,
            lora_alpha=rank * 2,
            lora_dropout=0.1,
            target_modules=["query", "value"],
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    return model


def build_training_args(args, run_name):
    common = dict(
        output_dir=os.path.join(args.output_dir, run_name),
        run_name=run_name,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        seed=args.seed,
        report_to=[],  # switch to ["wandb"] once a project is configured
    )

    if args.method == "galore":
        common.update(
            optim="galore_adamw",
            optim_target_modules=GALORE_TARGET_MODULES,
            optim_args=f"rank={args.rank}, update_proj_gap=200, scale={args.scale}",
        )
    # adamw and lora both use the Trainer's default AdamW optimizer;
    # LoRA's memory savings come from having fewer trainable parameters,
    # not from touching the optimizer.

    return TrainingArguments(**common)


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.method in ("lora", "galore") and args.rank is None:
        raise ValueError(f"--rank is required for --method {args.method}")

    run_name = args.method + (f"_r{args.rank}" if args.rank else "")
    if args.method == "galore" and args.scale != 0.25:
        run_name += f"_a{args.scale:g}"
    print(f"=== Running configuration: {run_name} ===")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds, eval_ds = load_data(tokenizer, args.max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = build_model(args.method, args.rank)
    training_args = build_training_args(args, run_name)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start = time.time()
    trainer.train()
    runtime_sec = time.time() - start

    metrics = trainer.evaluate()
    peak_vram_gb = (
        torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    )

    if args.save_model:
        save_dir = os.path.join("models", run_name)
        trainer.save_model(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Saved model to {save_dir}")

    print(f"Accuracy: {metrics['eval_accuracy']:.4f}")
    print(f"Peak VRAM: {peak_vram_gb:.2f} GB")
    print(f"Training time: {runtime_sec:.1f} sec")

    results_path = args.results_path
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    write_header = not os.path.exists(results_path)
    extended = results_path != RESULTS_PATH
    with open(results_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            cols = ["method", "rank", "seed"] + (["lr", "scale"] if extended else [])
            writer.writerow(cols + ["eval_accuracy", "peak_vram_gb", "train_runtime_sec"])
        row = [args.method, args.rank if args.rank else "-", args.seed]
        if extended:
            row += [args.lr, args.scale if args.method == "galore" else "-"]
        row += [f"{metrics['eval_accuracy']:.4f}", f"{peak_vram_gb:.3f}", f"{runtime_sec:.1f}"]
        writer.writerow(row)


if __name__ == "__main__":
    main()
