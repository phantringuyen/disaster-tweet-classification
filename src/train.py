"""
training script for Disaster Tweet Classification using HuggingFace models.
supports: DistilBERT, BERT-base, ModernBERT, RoBERTa.
"""

import argparse
import pandas as pd
from transformers import (
    Trainer, TrainingArguments, EarlyStoppingCallback,
    DistilBertTokenizerFast, DistilBertForSequenceClassification,
    BertTokenizerFast, BertForSequenceClassification,
    RobertaTokenizerFast, RobertaForSequenceClassification,
)
from datasets import Dataset
from utils import create_train_val_split, compute_metrics

# ModernBERT
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_model_and_tokenizer(model_name):
    """Return tokenizer and model depending on model_name."""

    if model_name == "distilbert":
        tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        )
    elif model_name == "bert":
        tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        )
    elif model_name == "modernbert":
        tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        model = AutoModelForSequenceClassification.from_pretrained(
            "answerdotai/ModernBERT-base", num_labels=2
        )
    elif model_name == "roberta":
        tok = RobertaTokenizerFast.from_pretrained("roberta-base")
        model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base", num_labels=2
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return tok, model


def tokenize_function(example, tokenizer):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=160,
    )


def main(args):
    print("\n=== Loading Dataset ===")
    df = pd.read_csv("data/train.csv")

    df_train, df_val = create_train_val_split(df)

    print("\n=== Loading Tokenizer & Model ===")
    tokenizer, model = load_model_and_tokenizer(args.model)

    # Convert to HF Dataset
    train_ds = Dataset.from_pandas(df_train)[["text", "target"]]
    val_ds   = Dataset.from_pandas(df_val)[["text", "target"]]

    train_ds = train_ds.rename_column("target", "labels")
    val_ds   = val_ds.rename_column("target", "labels")

    train_ds = train_ds.map(lambda e: tokenize_function(e, tokenizer), batched=True)
    val_ds   = val_ds.map(lambda e: tokenize_function(e, tokenizer), batched=True)

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    print("\n=== Training Setup ===")
    training_args = TrainingArguments(
        output_dir=f"results/model_checkpoints/{args.model}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\n=== Training… ===")
    trainer.train()

    print("\n=== Saving final model ===")
    trainer.save_model(f"results/model_checkpoints/{args.model}/final")
    tokenizer.save_pretrained(f"results/model_checkpoints/{args.model}/final")

    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bert",
                        choices=["distilbert", "bert", "modernbert", "roberta"],
                        help="Which model to train.")

    args = parser.parse_args()
    main(args)
