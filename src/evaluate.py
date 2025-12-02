"""
evaluation script for Disaster Tweet Classification.
loads a trained model checkpoint and evaluates on the validation split.
"""

import argparse
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from datasets import Dataset
from transformers import Trainer

from utils import create_train_val_split, compute_metrics
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    ticks = range(len(labels))
    plt.xticks(ticks, labels, rotation=45)
    plt.yticks(ticks, labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


def main(args):
    print("\n=== Loading Dataset ===")
    df = pd.read_csv("data/train.csv")
    df_train, df_val = create_train_val_split(df)

    print("\n=== Loading Model Checkpoint ===")
    model_path = f"results/model_checkpoints/{args.model}/final"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    val_ds = Dataset.from_pandas(df_val)[["text", "target"]]
    val_ds = val_ds.rename_column("target", "labels")

    val_ds = val_ds.map(lambda e: tokenizer(e["text"],
                                            truncation=True,
                                            padding="max_length",
                                            max_length=160),
                        batched=True)
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    trainer = Trainer(model=model, tokenizer=tokenizer)

    print("\n=== Predicting… ===")
    outputs = trainer.predict(val_ds)
    preds = outputs.predictions.argmax(-1)

    print("\n=== Classification Report ===")
    print(classification_report(df_val["target"], preds,
          target_names=["Not Disaster", "Disaster"]))

    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(df_val["target"], preds)
    plot_confusion_matrix(cm, ["Not Disaster", "Disaster"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bert",
                        choices=["distilbert", "bert", "modernbert", "roberta"],
                        help="Which model checkpoint to evaluate.")

    args = parser.parse_args()
    main(args)
