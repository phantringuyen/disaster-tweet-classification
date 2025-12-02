import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tabulate import tabulate
from sklearn.model_selection import train_test_split

def set_seed(seed=42):
    """
    set random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(train_path, test_path):
    """
    load train & test CSV files and return pandas DataFrames
    """
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    return df_train, df_test


def show_random_samples(df, n=5):
    """
    print random samples from a DataFrame in a table format
    """
    samples = df.sample(n)
    print(tabulate(samples, headers='keys', tablefmt='pretty', showindex=False))


def compute_metrics(eval_pred):
    """
    metric function for HuggingFace Trainer
    output: accuracy, precision, recall, F1
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def save_model(trainer, output_dir="saved_model"):
    """
    save model & tokenizer from HuggingFace trainer
    """
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    print(f"Model saved to: {output_dir}")


def load_pretrained_model(model_class, tokenizer_class, model_dir):
    """
    load model & tokenizer from folder
    """
    tokenizer = tokenizer_class.from_pretrained(model_dir)
    model = model_class.from_pretrained(model_dir)
    return model, tokenizer

def create_train_val_split(df, test_size=0.2, seed=42):
    """
    split into train and validation sets
    
    args:
        df: dataset
        test_size: percentage for validation
        seed : reproducibility

    output:
        df_train, df_val
    """
    df_train, df_val = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["target"]
    )
    
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    
    print(f"Training samples: {len(df_train)}")
    print(f"Validation samples: {len(df_val)}")

    return df_train, df_val
