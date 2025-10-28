"""Utilities for fine-tuning Hugging Face models on Kaggle style datasets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainingArguments

from .model_registry import load_model

MODELS_DIR = Path.home() / "credit-appraisal-agent-poc" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_kaggle_dataset(file_path: str) -> pd.DataFrame:
    """Load a Kaggle CSV/TSV dataset with basic delimiter detection."""

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    if path.suffix.lower() == ".tsv":
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Dataset {file_path} has no rows.")

    return df


def _encode_labels(df: pd.DataFrame, label_col: str) -> Dict[int, str]:
    """Ensure labels are numeric for transformers Trainer."""

    if pd.api.types.is_numeric_dtype(df[label_col]):
        return {}

    categories = df[label_col].astype("category")
    df[label_col] = categories.cat.codes
    return {int(code): str(label) for code, label in enumerate(categories.cat.categories)}


def _prepare_dataset(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    tokenizer,
    *,
    max_length: int = 512,
) -> Dataset:
    dataset = Dataset.from_pandas(df[[text_col, label_col]].copy())
    dataset = dataset.rename_column(label_col, "labels")

    def tokenize(batch: Dict[str, Iterable[str]]):
        return tokenizer(
            batch[text_col],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=[text_col])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


def train_agent(
    task_name: str,
    dataset_path: str,
    text_col: str,
    label_col: str,
    *,
    output_subdir: Optional[str] = None,
    num_train_epochs: int = 2,
    learning_rate: float = 2e-5,
    per_device_train_batch_size: int = 8,
    weight_decay: float = 0.01,
    warmup_steps: int = 0,
    max_train_samples: Optional[int] = None,
    evaluation_split: float = 0.2,
    seed: int = 42,
) -> Path:
    """Fine-tune the specified task on the provided Kaggle dataset."""

    tokenizer, model, _ = load_model(task_name)

    df = load_kaggle_dataset(dataset_path)
    if max_train_samples:
        df = df.head(max_train_samples)

    label_mapping = _encode_labels(df, label_col)
    if label_mapping:
        model.config.label2id = {v: k for k, v in label_mapping.items()}
        model.config.id2label = label_mapping
    model.config.num_labels = int(df[label_col].nunique())

    dataset = _prepare_dataset(df, text_col, label_col, tokenizer)
    if evaluation_split:
        split = dataset.train_test_split(test_size=evaluation_split, seed=seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    training_args = TrainingArguments(
        output_dir=str(MODELS_DIR / (output_subdir or task_name)),
        evaluation_strategy="epoch" if evaluation_split else "no",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        logging_dir=str(MODELS_DIR / "logs" / task_name),
        logging_steps=25,
        save_total_limit=2,
        load_best_model_at_end=bool(evaluation_split),
        metric_for_best_model="accuracy",
        greater_is_better=True,
        seed=seed,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted"),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics if evaluation_split else None,
    )

    trainer.train()

    output_dir = MODELS_DIR / f"{task_name}_trained" if output_subdir is None else MODELS_DIR / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metadata = {
        "task_name": task_name,
        "dataset_path": os.path.abspath(dataset_path),
        "text_col": text_col,
        "label_col": label_col,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": per_device_train_batch_size,
        "weight_decay": weight_decay,
    }
    if label_mapping:
        metadata["label_mapping"] = label_mapping
    with open(output_dir / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a sandbox agent on a Kaggle dataset")
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--text_col", required=True)
    parser.add_argument("--label_col", required=True)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--evaluation_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_subdir")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = train_agent(
        task_name=args.task_name,
        dataset_path=args.dataset_path,
        text_col=args.text_col,
        label_col=args.label_col,
        output_subdir=args.output_subdir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_train_samples=args.max_train_samples,
        evaluation_split=args.evaluation_split,
        seed=args.seed,
    )
    print(json.dumps({"status": "ok", "output_dir": str(output_dir)}))


if __name__ == "__main__":  # pragma: no cover
    main()
