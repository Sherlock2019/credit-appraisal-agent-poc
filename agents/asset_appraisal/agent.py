"""Asset appraisal agent used for collateral valuation."""
from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class AssetRecord:
    """Represents a stored asset valuation record."""

    data: Dict[str, Any]
    path: Path
    source: str


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return cleaned or "asset"


def _default_runs_dir() -> Path:
    override = os.getenv("ASSET_AGENT_RUNS_ROOT")
    if override:
        path = Path(override).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    return Path(__file__).resolve().parents[2] / "services" / "api" / ".runs" / "asset"


class AssetAppraisalAgent:
    """Simple asset appraisal agent that persists valuation runs."""

    def __init__(self, runs_root: Optional[str | os.PathLike[str]] = None) -> None:
        self.model_name = "AssetValNet-v1"
        base_dir = Path(runs_root).expanduser().resolve() if runs_root else _default_runs_dir()
        base_dir.mkdir(parents=True, exist_ok=True)
        self.base_dir = base_dir
        self.valuations_dir = self.base_dir / "valuations"
        self.verified_dir = self.base_dir / "verified"
        self.field_dir = self.base_dir / "field_data"
        self.valuations_dir.mkdir(parents=True, exist_ok=True)
        self.verified_dir.mkdir(parents=True, exist_ok=True)
        self.field_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate(self, asset_type: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        slug = _slugify(asset_type)
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        asset_id = f"valuation_{slug}_{timestamp}"

        declared_value = self._safe_float(metadata.get("declared_value"))
        base_value = declared_value if declared_value else random.uniform(75_000, 950_000)
        market_factor = random.uniform(0.9, 1.1)
        risk_factor = random.uniform(0.85, 1.15)
        estimated_value = base_value * market_factor * risk_factor

        result = {
            "asset_id": asset_id,
            "asset_type": asset_type,
            "metadata": metadata or {},
            "estimated_value": round(estimated_value, 2),
            "confidence": round(random.uniform(0.82, 0.97), 2),
            "model_name": self.model_name,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "valuation",
        }

        output_path = self.valuations_dir / f"{asset_id}.json"
        self._write_json(output_path, result)
        return result

    def apply_verification(
        self,
        asset_id: str,
        verified: bool,
        legitimacy_score: float,
        inspector_notes: str = "",
        local_authority_ref: str = "",
    ) -> Dict[str, Any]:
        record = self._load_valuation(asset_id)
        if record is None:
            raise FileNotFoundError(f"No valuation record found for asset_id '{asset_id}'.")

        data = record.data
        legitimacy_score = max(0.0, min(float(legitimacy_score), 1.0))
        base_confidence = self._safe_float(data.get("confidence"), 0.9)

        if verified:
            confidence = min(1.0, base_confidence * (0.85 + legitimacy_score * 0.15))
        else:
            confidence = max(0.0, base_confidence * 0.5)

        data.update(
            {
                "verified": bool(verified),
                "legitimacy_score": round(legitimacy_score, 3),
                "inspector_notes": inspector_notes,
                "local_authority_ref": local_authority_ref,
                "verified_timestamp": datetime.utcnow().isoformat(),
                "confidence": round(confidence, 2),
            }
        )

        latest_field = self._latest_field_data(asset_id)
        if latest_field:
            data["field_data"] = latest_field

        data["source"] = "verified" if verified else data.get("source", "valuation")

        verified_path = self.verified_dir / f"verified_{asset_id}_{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.json"
        self._write_json(verified_path, data)
        return data

    def store_field_data(
        self,
        asset_id: str,
        inspector_name: str,
        latitude: float,
        longitude: float,
        notes: str = "",
        photo_bytes: Optional[bytes] = None,
        photo_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        meta = {
            "asset_id": asset_id,
            "inspector_name": inspector_name,
            "timestamp": timestamp,
            "gps": {"latitude": float(latitude), "longitude": float(longitude)},
            "notes": notes,
        }

        photo_path = None
        if photo_bytes:
            extension = Path(photo_name or "field_photo").suffix or ".jpg"
            photo_filename = f"{asset_id}_{timestamp}{extension}"
            photo_path = self.field_dir / photo_filename
            photo_path.write_bytes(photo_bytes)
            meta["photo"] = {"filename": photo_filename, "path": str(photo_path)}

        meta_path = self.field_dir / f"{asset_id}_{timestamp}.json"
        self._write_json(meta_path, meta)

        response = {"status": "stored", "meta_path": str(meta_path)}
        if photo_path:
            response["photo_path"] = str(photo_path)
        response["data"] = meta
        return response

    def get_latest_asset_record(self) -> Optional[Dict[str, Any]]:
        for directory, source in ((self.verified_dir, "verified"), (self.valuations_dir, "valuation")):
            record = self._latest_from_dir(directory, source)
            if record:
                payload = dict(record.data)
                payload.setdefault("source", source)
                return payload
        return None

    # Backwards compatibility helpers
    def get_latest_verified_asset(self) -> Optional[Tuple[float, float, float]]:
        record = self._latest_from_dir(self.verified_dir, "verified")
        if record:
            data = record.data
            return (
                self._safe_float(data.get("estimated_value"), 0.0),
                self._safe_float(data.get("confidence"), 0.0),
                self._safe_float(data.get("legitimacy_score"), 0.0),
            )
        return None

    def get_latest_valuation(self) -> Optional[Tuple[float, float, float]]:
        record = self._latest_from_dir(self.valuations_dir, "valuation")
        if record:
            data = record.data
            return (
                self._safe_float(data.get("estimated_value"), 0.0),
                self._safe_float(data.get("confidence"), 0.0),
                self._safe_float(data.get("legitimacy_score"), 0.0),
            )
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def _load_valuation(self, asset_id: str) -> Optional[AssetRecord]:
        direct_path = self.valuations_dir / f"{asset_id}.json"
        if direct_path.exists():
            return AssetRecord(self._read_json(direct_path), direct_path, "valuation")

        matches = sorted(self.valuations_dir.glob(f"{asset_id}*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in matches:
            return AssetRecord(self._read_json(path), path, "valuation")
        return None

    def _latest_from_dir(self, directory: Path, source: str) -> Optional[AssetRecord]:
        if not directory.exists():
            return None
        files = sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in files:
            try:
                data = self._read_json(path)
                data.setdefault("source", source)
                return AssetRecord(data, path, source)
            except Exception:
                continue
        return None

    def _latest_field_data(self, asset_id: str) -> Optional[Dict[str, Any]]:
        files = sorted(self.field_dir.glob(f"{asset_id}*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in files:
            try:
                data = self._read_json(path)
                data.setdefault("source", "field_data")
                return data
            except Exception:
                continue
        return None

    def _read_json(self, path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _safe_float(value: Any, default: float | None = None) -> float | None:
        try:
            if value is None:
                return default
            return float(value)
        except Exception:
            return default
