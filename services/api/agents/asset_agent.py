"""Asset appraisal agent combining text and image encoders."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from PIL import Image

from .model_registry import load_model


@dataclass
class AssetAppraisalResult:
    """Structured output for property valuation estimates."""

    valuation_score: float
    risk_score: float
    notes: str


class AssetAppraisalAgent:
    """Estimate asset value using text descriptions and optional imagery."""

    def __init__(self) -> None:
        self.text_tokenizer, self.text_model = load_model("asset_appraisal_text")
        self.image_processor, self.image_model = load_model("asset_appraisal_image")
        if self.text_tokenizer is None:
            raise RuntimeError("Asset appraisal text model must expose a tokenizer")
        self.text_model.eval()
        if self.image_model is not None:
            self.image_model.eval()

    @torch.inference_mode()
    def _encode_text(self, descriptions: Iterable[str]) -> torch.Tensor:
        tokens = self.text_tokenizer(list(descriptions), return_tensors="pt", padding=True, truncation=True)
        outputs = self.text_model(**tokens)
        # Use CLS token as representation when available, otherwise mean pooling
        if hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state
            if hidden.size(1) > 1:
                return hidden[:, 0, :]
            return hidden.mean(dim=1)
        if hasattr(outputs, "logits"):
            return outputs.logits
        raise RuntimeError("Unexpected output from text model")

    @torch.inference_mode()
    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        if self.image_processor is None or self.image_model is None:
            raise RuntimeError("Vision backbone is not configured for asset appraisal")
        processed = self.image_processor(images=image, return_tensors="pt")
        outputs = self.image_model(**processed)
        if hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state
            return hidden.mean(dim=1)
        if hasattr(outputs, "pooler_output"):
            return outputs.pooler_output
        raise RuntimeError("Unexpected output from vision model")

    @torch.inference_mode()
    def appraise(
        self,
        description: str,
        image: Optional[Image.Image] = None,
    ) -> AssetAppraisalResult:
        """Generate a coarse valuation and risk score."""

        text_embedding = self._encode_text([description])[0]
        text_score = torch.sigmoid(text_embedding.mean()).item()

        if image is not None:
            image_embedding = self._encode_image(image)[0]
            image_score = torch.sigmoid(image_embedding.mean()).item()
        else:
            image_score = 0.5

        valuation_score = float((text_score * 0.6) + (image_score * 0.4))
        risk_score = float(1.0 - valuation_score)

        note_parts = [
            f"Text confidence={text_score:.2f}",
            f"Image confidence={image_score:.2f}" if image is not None else "Image unavailable",
        ]
        notes = "; ".join(note_parts)
        return AssetAppraisalResult(
            valuation_score=valuation_score,
            risk_score=risk_score,
            notes=notes,
        )
