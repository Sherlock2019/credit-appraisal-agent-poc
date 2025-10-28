"""Central Hugging Face model registry used by sandbox agents."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoProcessor,
    AutoTokenizer,
    VisionEncoderDecoderModel,
)

try:  # Optional import depending on transformers version
    from transformers import AutoImageProcessor
except ImportError:  # pragma: no cover - fallback for older releases
    AutoImageProcessor = AutoProcessor  # type: ignore


@dataclass(frozen=True)
class TaskConfig:
    """Metadata describing how to bootstrap a Hugging Face checkpoint."""

    model_id: str
    task_type: str = "sequence_classification"
    revision: Optional[str] = None


MODEL_MAP: Dict[str, TaskConfig] = {
    "credit_appraisal": TaskConfig("roberta-base", task_type="sequence_classification"),
    "asset_appraisal_text": TaskConfig("distilbert-base-uncased", task_type="sequence_classification"),
    "asset_appraisal_image": TaskConfig("google/vit-base-patch16-224", task_type="image_classification"),
    "kyc_text": TaskConfig("microsoft/layoutlm-base-uncased", task_type="token_classification"),
    "kyc_ocr": TaskConfig("microsoft/trocr-base-stage1", task_type="vision_to_text"),
    "fraud_detection": TaskConfig("bert-base-uncased", task_type="sequence_classification"),
    "customer_support": TaskConfig("distilbert-base-uncased", task_type="sequence_classification"),
}


def list_registered_tasks() -> Dict[str, str]:
    """Return a simplified mapping of task -> model id for API responses."""

    return {task: cfg.model_id for task, cfg in MODEL_MAP.items()}


@lru_cache(maxsize=10)
def load_model(task_name: str, *, model_id: Optional[str] = None):
    """Load a tokenizer/model pair for a registered task.

    Parameters
    ----------
    task_name:
        Key defined in :data:`MODEL_MAP`.
    model_id:
        Optional override to load a custom checkpoint.

    Returns
    -------
    tuple(tokenizer, model, processor)
        Tokenizer is ``None`` for pure vision models, processor is ``None`` for
        classic text models. Consumers can safely ignore whichever value is not
        required for their task type.
    """

    if task_name not in MODEL_MAP:
        raise ValueError(f"No model registered for task '{task_name}'.")

    cfg = MODEL_MAP[task_name]
    checkpoint = model_id or cfg.model_id
    revision = cfg.revision

    tokenizer = None
    processor = None

    if cfg.task_type == "image_classification":
        processor = AutoImageProcessor.from_pretrained(checkpoint, revision=revision)
        model = AutoModelForImageClassification.from_pretrained(checkpoint, revision=revision)
    elif cfg.task_type == "vision_to_text":
        processor = AutoProcessor.from_pretrained(checkpoint, revision=revision)
        model = VisionEncoderDecoderModel.from_pretrained(checkpoint, revision=revision)
    elif cfg.task_type == "token_classification":
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
        model = AutoModelForTokenClassification.from_pretrained(checkpoint, revision=revision)
    elif cfg.task_type == "causal_lm":
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(checkpoint, revision=revision)
    elif cfg.task_type == "encoder":
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
        model = AutoModel.from_pretrained(checkpoint, revision=revision)
    else:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, revision=revision)

    return tokenizer, model, processor


__all__ = ["MODEL_MAP", "TaskConfig", "list_registered_tasks", "load_model"]
