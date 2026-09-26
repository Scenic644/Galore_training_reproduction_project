# src/train.py
import yaml
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score

def load_sst2():
    dataset = load_dataset("glue", "sst2")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    def tokenize(examples):
        return tokenizer(examples["sentence"],
                         truncation=True, padding="max_length",
                         max_length=128)
    dataset = dataset.map(tokenize, batched=True)
    return dataset, tokenizer

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

def train(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dataset, tokenizer = load_sst2()

    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=2
    )

    # Configure optimizer
    if cfg["method"] == "adamw":
        optim = "adamw_torch"
        optim_args = None
        optim_target = None
    elif cfg["method"] == "lora":
        optim = "adamw_torch"
        optim_args = None
        optim_target = None
        # Apply LoRA via peft
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=cfg["rank"], lora_alpha=16, lora_dropout=0.1,
            target_modules=["query", "value"],
        )
        model = get_peft_model(model, lora_config)
    elif cfg["method"] == "galore":
    optim = "galore_adamw"
    optim_target = ["query", "key", "value", "dense", "intermediate", "output"]
    optim_args = (f"rank={cfg['rank']},"
                  f"update_proj_gap={cfg['update_proj_gap']},"
                  f"scale={cfg['scale']}")

    training_args = TrainingArguments(
        output_dir=f"./results/{cfg['run_name']}",
        learning_rate=cfg["lr"],
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=cfg["epochs"],
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        optim=optim,
        optim_target_modules=optim_target,
        optim_args=optim_args,
        logging_steps=50,
        report_to="none",  # or "wandb" if you set it up
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
    )

    # Measure peak memory before training
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    trainer.train()

    # Record peak memory
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Peak GPU memory: {peak_mem_mb:.1f} MB")
        # Save to results
        with open(f"./results/{cfg['run_name']}/peak_memory.txt", "w") as f:
            f.write(str(peak_mem_mb))

    # Final evaluation
    results = trainer.evaluate()
    print(f"Final accuracy: {results['eval_accuracy']:.4f}")

if __name__ == "__main__":
    import sys
    train(sys.argv[1])
