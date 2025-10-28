"""KYC helper utilities using LayoutLM and TrOCR checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import torch
from PIL import Image
from transformers import pipeline

from .model_registry import load_model


def _resolve_torch_device(device: str | int | None) -> torch.device:
    if device is None:
        return torch.device("cpu")
    if isinstance(device, int):
        if device < 0:
            return torch.device("cpu")
        return torch.device(f"cuda:{device}")
    return torch.device(device)


@dataclass
class KYCAgent:
    text_task: str = "kyc_text"
    ocr_task: str = "kyc_ocr"
    device: str | int | None = None
    aggregation_strategy: str = "simple"
    _ner_pipeline: Any = field(init=False, repr=False)
    _ocr_model: Any = field(init=False, repr=False)
    _ocr_processor: Any = field(init=False, repr=False)
    _torch_device: torch.device = field(init=False, repr=False)

    def __post_init__(self) -> None:
        tokenizer, model, _ = load_model(self.text_task)
        self._ner_pipeline = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            device=self.device,
            aggregation_strategy=self.aggregation_strategy,
        )

        _, ocr_model, processor = load_model(self.ocr_task)
        if processor is None:
            raise RuntimeError("OCR task requires a processor but none was loaded.")
        self._torch_device = _resolve_torch_device(self.device)
        self._ocr_model = ocr_model.to(self._torch_device)
        self._ocr_model.eval()
        self._ocr_processor = processor

    # ────────────────────────────
    # Text utilities
    # ────────────────────────────
    def extract_entities(self, text: str) -> List[Any]:
        """Extract structured entities from a single KYC document."""

        return self._ner_pipeline(text)

    def batch_extract_entities(self, texts: Sequence[str]) -> List[List[Any]]:
        if not isinstance(texts, (list, tuple)):
            raise TypeError("texts must be a sequence of strings")
        return [self._ner_pipeline(t) for t in texts]

    # ────────────────────────────
    # OCR utilities
    # ────────────────────────────
    def read_document(self, image: str | Path | Image.Image) -> str:
        img = self._ensure_image(image)
        inputs = self._ocr_processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._torch_device)
        with torch.no_grad():
            generated = self._ocr_model.generate(pixel_values)
        text = self._ocr_processor.batch_decode(generated, skip_special_tokens=True)
        return text[0] if text else ""

    def batch_read_documents(self, images: Iterable[str | Path | Image.Image]) -> List[str]:
        return [self.read_document(img) for img in images]

    @staticmethod
    def _ensure_image(image: str | Path | Image.Image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        with Image.open(path) as img:
            return img.convert("RGB")


__all__ = ["KYCAgent"]
