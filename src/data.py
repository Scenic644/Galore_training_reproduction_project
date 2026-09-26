"""
SST-2 (GLUE) data loading and tokenization for RoBERTa-base.

Written by: Muhammad Umer, Duaa Naz, Reham Hafeez
"""
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "roberta-base"
MAX_LENGTH = 128


def load_sst2():
    """Load SST-2 from the GLUE benchmark and tokenize it for RoBERTa-base.

    Returns
    -------
    dataset : DatasetDict with 'train' and 'validation' splits, tokenized
              and ready to hand to a Trainer (columns: input_ids,
              attention_mask, labels).
    tokenizer : the RoBERTa tokenizer used.
    """
    load_dataset("nyu-mll/glue", "sst2")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    dataset = dataset.map(tokenize, batched=True)

    # GLUE's column is called "label" (singular). AutoModelForSequenceClassification's
    # forward() expects "labels" (plural). Without this rename, Trainer's default
    # remove_unused_columns=True silently drops the label column and training fails
    # with "did not return a loss".
    dataset = dataset.remove_columns(["sentence", "idx"])
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch")

    return dataset, tokenizer
