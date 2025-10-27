"""Credit appraisal agent powered by a Hugging Face sequence classifier."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch

from .model_registry import load_model


@dataclass
class CreditAssessment:
    """Structured output for a credit application inference."""

    label: int
    confidence: float
    probabilities: List[float]


class CreditAppraisalAgent:
    """Thin wrapper around a pretrained credit scoring model.

    The class is intentionally lightweight: it focuses on orchestrating the
    tokenizer/model duo and exposes convenience helpers for single or batched
    inference.  Consumers are expected to provide their own domain-specific
    post-processing on top of the raw probabilities.
    """

    def __init__(self, task_name: str = "credit_appraisal") -> None:
        self.tokenizer, self.model = load_model(task_name)
        if self.tokenizer is None:
            raise RuntimeError(
                "The registered model does not expose a tokenizer – please verify"
                f" task '{task_name}'."
            )
        self.model.eval()

    @torch.inference_mode()
    def predict(self, text: str) -> CreditAssessment:
        """Run inference for a single applicant description."""

        encoded = self.tokenizer(text, return_tensors="pt", truncation=True)
        output = self.model(**encoded)
        probs = torch.softmax(output.logits, dim=-1).squeeze(0)
        confidence, label = torch.max(probs, dim=-1)
        return CreditAssessment(
            label=int(label.item()),
            confidence=float(confidence.item()),
            probabilities=probs.tolist(),
        )

    @torch.inference_mode()
    def batch_predict(self, texts: Sequence[str]) -> List[CreditAssessment]:
        """Run inference for a batch of applicant descriptions."""

        if not texts:
            return []
        encoded = self.tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True)
        output = self.model(**encoded)
        probs = torch.softmax(output.logits, dim=-1)
        labels = torch.argmax(probs, dim=-1)
        confidences = torch.gather(probs, 1, labels.unsqueeze(1)).squeeze(1)
        return [
            CreditAssessment(
                label=int(label.item()),
                confidence=float(conf.item()),
                probabilities=prob.tolist(),
            )
            for prob, label, conf in zip(probs, labels, confidences)
        ]
