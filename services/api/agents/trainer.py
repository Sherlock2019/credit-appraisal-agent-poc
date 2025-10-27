"""Training utilities for fine-tuning Hugging Face models with Kaggle datasets."""
from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from .model_registry import MODEL_MAP, load_model


TRAINABLE_TASKS = {
    "credit_appraisal",
    "asset_appraisal_text",
    "kyc_text",
    "fraud_detection",
    "customer_support",
}

MODELS_DIR = os.path.expanduser("~/credit-appraisal-agent-poc/models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_kaggle_dataset(file_path: str) -> pd.DataFrame:
    """Load a Kaggle CSV/TSV dataset."""

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"Dataset {file_path} is empty")
    return df


def _prepare_labels(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, Dict[int, str]]:
    unique_labels = sorted(df[label_col].dropna().unique().tolist())
    if not unique_labels:
        raise ValueError(f"Column '{label_col}' contains no labels")
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    df = df.copy()
    df["label_id"] = df[label_col].map(label2id)
    if df["label_id"].isna().any():
        raise ValueError(f"Unable to map some labels in column '{label_col}'")
    id2label = {idx: str(label) for label, idx in label2id.items()}
    return df, id2label


def train_agent(task_name: str, dataset_path: str, text_col: str, label_col: str) -> str:
    """Fine-tune a Hugging Face sequence classifier on a Kaggle dataset."""

    if task_name not in TRAINABLE_TASKS:
        raise ValueError(
            f"Task '{task_name}' is not configured for text fine-tuning. Supported: {sorted(TRAINABLE_TASKS)}"
        )

    tokenizer, model = load_model(task_name)
    if tokenizer is None:
        raise RuntimeError(f"Task '{task_name}' does not expose a tokenizer suitable for text training")
    if not isinstance(model, AutoModelForSequenceClassification):
        # Reload the model explicitly as a classifier – useful for embeddings-only registries.
        model_name = getattr(model, "name_or_path", None) or MODEL_MAP.get(task_name, "roberta-base")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

    df = load_kaggle_dataset(dataset_path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Dataset must contain columns '{text_col}' and '{label_col}'")

    df = df.dropna(subset=[text_col, label_col])
    if df.empty:
        raise ValueError("Dataset has no rows after dropping missing text/label values")

    df, id2label = _prepare_labels(df, label_col)
    label2id = {label: idx for idx, label in id2label.items()}

    dataset = Dataset.from_pandas(df[[text_col, "label_id"]], preserve_index=False)

    def tokenize(batch):
        tokens = tokenizer(batch[text_col], padding=True, truncation=True)
        tokens["labels"] = batch["label_id"]
        return tokens

    tokenized = dataset.map(tokenize, batched=True, remove_columns=[text_col, "label_id"])
    tokenized.set_format(type="torch")

    if len(tokenized) > 1:
        splits = tokenized.train_test_split(test_size=0.2, seed=42)
        train_dataset = splits["train"]
        eval_dataset = splits["test"]
    else:
        train_dataset = tokenized
        eval_dataset = tokenized

    training_args = TrainingArguments(
        output_dir=os.path.join(MODELS_DIR, task_name),
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_dir=os.path.join(MODELS_DIR, "logs"),
        save_total_limit=1,
    )

    model.config.label2id = {str(k): int(v) for k, v in label2id.items()}
    model.config.id2label = {int(k): str(v) for k, v in id2label.items()}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    save_path = os.path.join(MODELS_DIR, f"{task_name}_trained")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    return save_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Hugging Face models on Kaggle datasets")
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--text_col", required=True)
    parser.add_argument("--label_col", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    save_path = train_agent(
        task_name=args.task_name,
        dataset_path=args.dataset_path,
        text_col=args.text_col,
        label_col=args.label_col,
    )
    print(f"✅ Saved fine-tuned model to {save_path}")


if __name__ == "__main__":
    main()
