"""
Training script for the GaLore reproduction (AdamW baseline, LoRA, and
GaLore at several ranks) on RoBERTa-base / SST-2.

Written by: Muhammad Umer, Duaa Naz, Reham Hafeez

Usage:
    python src/train.py configs/adamw_baseline.yaml            # full run
    python src/train.py configs/adamw_baseline.yaml --sanity   # 1-epoch, tiny
                                                                # subset, to
                                                                # check the
                                                                # pipeline
                                                                # before a
                                                                # real run
"""
import json
import os
import sys

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_sst2  # noqa: E402

MODEL_NAME = "roberta-base"

# RoBERTa's linear layers are named query / key / value (self-attention) and
# dense (attention output, intermediate, and final feed-forward layers).
# GaLore's own examples default to ["attn", "mlp"], which are LLaMA-style
# names and match nothing in RoBERTa. Using those defaults here would make
# GaLore silently apply to zero parameters.
GALORE_TARGET_MODULES = ["query", "key", "value", "dense"]


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}


def build_model_and_optim(cfg):
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    optim = "adamw_torch"
    optim_target_modules = None
    optim_args = None

    if cfg["method"] == "adamw":
        pass  # defaults above are correct for the full fine-tuning baseline

    elif cfg["method"] == "lora":
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=cfg["rank"],
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["query", "value"],
        )
        model = get_peft_model(model, lora_config)

    elif cfg["method"] == "galore":
        optim = "galore_adamw"
        optim_target_modules = GALORE_TARGET_MODULES
        optim_args = (
            f"rank={cfg['rank']},"
            f"update_proj_gap={cfg['update_proj_gap']},"
            f"scale={cfg['scale']}"
        )

    else:
        raise ValueError(f"Unknown method in config: {cfg['method']}")

    return model, optim, optim_target_modules, optim_args


def train(config_path, sanity_check=False):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # YAML only parses "2e-5" as a float if it has a decimal point (e.g. "2.0e-5").
    # Written as "2e-5" it silently loads as a string. Cast here so an edited
    # config can't reintroduce that bug.
    cfg["lr"] = float(cfg["lr"])

    dataset, _ = load_sst2()

    if sanity_check:
        print("Running in --sanity mode: 1 epoch, tiny data subset.")
        dataset["train"] = dataset["train"].select(range(200))
        dataset["validation"] = dataset["validation"].select(range(50))
        cfg = {**cfg, "epochs": 1}

    model, optim, optim_target_modules, optim_args = build_model_and_optim(cfg)

    output_dir = f"./results/{cfg['run_name']}"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=cfg["lr"],
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=cfg["epochs"],
        weight_decay=0.01,
        eval_strategy="epoch",  # NOT evaluation_strategy: removed in transformers>=4.46
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        optim=optim,
        optim_target_modules=optim_target_modules,
        optim_args=optim_args,
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    trainer.train()

    peak_mem_mb = None
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Peak GPU memory: {peak_mem_mb:.1f} MB")

    results = trainer.evaluate()
    print(f"Final accuracy: {results['eval_accuracy']:.4f}")

    if not sanity_check:
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(
                {
                    "run_name": cfg["run_name"],
                    "method": cfg["method"],
                    "rank": cfg.get("rank"),
                    "accuracy": results["eval_accuracy"],
                    "peak_memory_mb": peak_mem_mb,
                },
                f,
                indent=2,
            )
        print(f"Saved metrics to {output_dir}/metrics.json")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/train.py <config path> [--sanity]")
        sys.exit(1)
    config_path = sys.argv[1]
    sanity = "--sanity" in sys.argv
    train(config_path, sanity_check=sanity)
