"""High level wrapper around the Hugging Face credit scoring model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Sequence

from transformers import pipeline

from .model_registry import load_model


@dataclass
class CreditAppraisalTextAgent:
    """Utility wrapper that exposes text classification for credit scoring."""

    task_name: str = "credit_appraisal"
    device: str | int | None = None
    _pipeline: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        tokenizer, model, _ = load_model(self.task_name)
        self._pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=self.device,
            return_all_scores=True,
        )

    def score_texts(self, texts: Sequence[str], *, top_k: int | None = None) -> List[Any]:
        """Score a list of applicant narratives using the configured model."""

        if not isinstance(texts, (list, tuple)):
            raise TypeError("texts must be a sequence of strings")
        if not texts:
            return []

        outputs = self._pipeline(list(texts), truncation=True)
        if top_k is None:
            return outputs
        truncated: List[Any] = []
        for record in outputs:
            if isinstance(record, list):
                truncated.append(sorted(record, key=lambda r: r.get("score", 0.0), reverse=True)[:top_k])
            else:
                truncated.append(record)
        return truncated

    def score_single(self, text: str, *, top_k: int | None = None) -> Any:
        """Convenience wrapper for a single text snippet."""

        return self.score_texts([text], top_k=top_k)[0]


__all__ = ["CreditAppraisalTextAgent"]
