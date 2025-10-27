"""Know-your-customer agent for OCR and document risk scoring."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from .model_registry import load_model


@dataclass
class KycResult:
    """Structured response for KYC document checks."""

    extracted_text: str
    risk_label: int
    confidence: float


class KYCAgent:
    """Performs OCR followed by document classification."""

    def __init__(self) -> None:
        self.text_tokenizer, self.text_model = load_model("kyc_text")
        self.ocr_processor, self.ocr_model = load_model("kyc_ocr")
        if self.text_tokenizer is None:
            raise RuntimeError("KYC text model requires a tokenizer")
        self.text_model.eval()
        if self.ocr_model is not None:
            self.ocr_model.eval()

    @torch.inference_mode()
    def extract_text(self, image) -> str:
        if self.ocr_processor is None or self.ocr_model is None:
            raise RuntimeError("OCR backbone is not configured for KYC agent")
        processed = self.ocr_processor(images=image, return_tensors="pt")
        generated = self.ocr_model.generate(**processed)
        texts = self.ocr_processor.batch_decode(generated, skip_special_tokens=True)
        return texts[0] if texts else ""

    @torch.inference_mode()
    def classify(self, text: str) -> Dict[str, float]:
        encoded = self.text_tokenizer(text, return_tensors="pt", truncation=True)
        outputs = self.text_model(**encoded)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
        return {str(i): float(p) for i, p in enumerate(probs.tolist())}

    @torch.inference_mode()
    def process_document(self, image) -> KycResult:
        extracted_text = self.extract_text(image)
        probabilities = self.classify(extracted_text)
        if probabilities:
            risk_label = max(probabilities, key=probabilities.get)
            confidence = probabilities[risk_label]
        else:
            risk_label, confidence = "0", 0.0
        return KycResult(
            extracted_text=extracted_text,
            risk_label=int(risk_label),
            confidence=float(confidence),
        )
