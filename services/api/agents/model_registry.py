"""Model registry for Hugging Face integrations.

This module lazily loads tokenizers/processors and models for the sandbox
agents.  The registry is intentionally small to keep downloads manageable while
providing sensible defaults that match the published architecture overview.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Tuple, Union

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModel,
    AutoImageProcessor,
    AutoProcessor,
    VisionEncoderDecoderModel,
)

# Mapping between logical task names and Hugging Face model identifiers.
MODEL_MAP = {
    "credit_appraisal": "roberta-base",
    "asset_appraisal_text": "distilbert-base-uncased",
    "asset_appraisal_image": "google/vit-base-patch16-224",
    "kyc_text": "microsoft/layoutlm-base-uncased",
    "kyc_ocr": "microsoft/trocr-base-stage1",
    "fraud_detection": "bert-base-uncased",
    "customer_support": "distilbert-base-uncased",
}

ProcessorType = Union[AutoTokenizer, AutoImageProcessor, AutoProcessor]


@lru_cache(maxsize=10)
def load_model(task_name: str) -> Tuple[ProcessorType | None, object]:
    """Load the processor/tokenizer and model for a registered task.

    Parameters
    ----------
    task_name:
        Logical task identifier.  See :data:`MODEL_MAP` for the list of
        supported tasks.

    Returns
    -------
    Tuple[processor, model]
        A Hugging Face processor (tokenizer/image processor) paired with the
        model instance.  Vision models may return ``None`` for the processor if
        the upstream checkpoint does not expose one.

    Raises
    ------
    ValueError
        If ``task_name`` is not registered in :data:`MODEL_MAP`.
    """

    model_id = MODEL_MAP.get(task_name)
    if not model_id:
        raise ValueError(f"No model registered for task '{task_name}'")

    # Heuristic detection of model families.
    lowered = model_id.lower()
    if "vit" in lowered:
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
    elif "trocr" in lowered:
        processor = AutoProcessor.from_pretrained(model_id)
        model = VisionEncoderDecoderModel.from_pretrained(model_id)
    elif "gpt" in lowered or "lm" in lowered:
        processor = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
    else:
        processor = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)

    return processor, model
