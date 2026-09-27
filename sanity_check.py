"""
sanity_check.py -- run this once before launching any full training run.

Loads a tiny slice of SST-2, pushes eight examples through RoBERTa-base,
and confirms the output shape and loss look right. Catches shape bugs and
data problems in seconds instead of after an overnight run.
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

MODEL_NAME = "roberta-base"

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

print("Loading a small slice of SST-2...")
raw = load_dataset("nyu-mll/glue", "sst2", split="train[:16]")
tokenized = raw.map(
    lambda batch: tokenizer(batch["sentence"], truncation=True, max_length=128),
    batched=True,
)
tokenized = tokenized.rename_column("label", "labels")
tokenized = tokenized.remove_columns(["sentence", "idx"])

collator = DataCollatorWithPadding(tokenizer=tokenizer)
batch = collator([tokenized[i] for i in range(8)])

print("Running one forward pass...")
model.eval()
with torch.no_grad():
    outputs = model(**batch)

print("Logits shape:", tuple(outputs.logits.shape))  # expect (8, 2)
print("Loss:", outputs.loss.item())                   # expect a small finite number

assert outputs.logits.shape == (8, 2), "Unexpected logits shape -- check num_labels."
assert torch.isfinite(outputs.loss), "Loss is not finite -- check inputs and labels."

print("Sanity check passed. Safe to launch a full training run.")
