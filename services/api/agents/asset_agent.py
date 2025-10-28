"""Asset appraisal helper built on top of Hugging Face models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Sequence

from PIL import Image
from transformers import pipeline

from .model_registry import load_model


@dataclass
class AssetAppraisalAgent:
    """Agent that combines text + image signals for asset valuation."""

    text_task: str = "asset_appraisal_text"
    image_task: str = "asset_appraisal_image"
    device: str | int | None = None
    _text_pipeline: Any = field(init=False, repr=False)
    _image_pipeline: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        tokenizer, model, _ = load_model(self.text_task)
        self._text_pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=self.device,
            return_all_scores=True,
        )

        _, image_model, processor = load_model(self.image_task)
        self._image_pipeline = pipeline(
            "image-classification",
            model=image_model,
            feature_extractor=processor,
            device=self.device,
        )

    def score_descriptions(self, descriptions: Sequence[str]) -> List[Any]:
        if not isinstance(descriptions, (list, tuple)):
            raise TypeError("descriptions must be a sequence of strings")
        if not descriptions:
            return []
        return self._text_pipeline(list(descriptions), truncation=True)

    def score_images(self, images: Iterable[str | Path | Image.Image]) -> List[Any]:
        prepared: List[Image.Image] = []
        for item in images:
            if isinstance(item, Image.Image):
                prepared.append(item.convert("RGB"))
            else:
                path = Path(item)
                if not path.exists():
                    raise FileNotFoundError(f"Image not found: {path}")
                with Image.open(path) as img:
                    prepared.append(img.convert("RGB"))
        if not prepared:
            return []
        return self._image_pipeline(prepared)


__all__ = ["AssetAppraisalAgent"]
