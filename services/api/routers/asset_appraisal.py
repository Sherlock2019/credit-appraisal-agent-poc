"""Routes for the asset appraisal agent."""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from agents.asset_appraisal import AssetAppraisalAgent

router = APIRouter(prefix="/v1/agents/asset_appraisal", tags=["asset_appraisal"])


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


@router.post("/run")
def run_asset_appraisal(asset_type: str = Form(...), metadata: str = Form("{}")) -> Dict[str, Any]:
    try:
        parsed_meta = json.loads(metadata) if metadata else {}
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid metadata JSON: {exc}") from exc

    try:
        agent = AssetAppraisalAgent()
        result = agent.evaluate(asset_type, parsed_meta)
        return {"status": "ok", "result": result}
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Asset appraisal failed: {exc}") from exc


@router.post("/verify")
def verify_asset_legitimacy(
    asset_id: str = Form(...),
    verified: Optional[str] = Form(...),
    legitimacy_score: float = Form(...),
    inspector_notes: str = Form(""),
    local_authority_ref: str = Form(""),
) -> Dict[str, Any]:
    agent = AssetAppraisalAgent()
    try:
        updated = agent.apply_verification(
            asset_id=asset_id,
            verified=_parse_bool(verified),
            legitimacy_score=legitimacy_score,
            inspector_notes=inspector_notes,
            local_authority_ref=local_authority_ref,
        )
        return {"status": "ok", "result": updated}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Verification failed: {exc}") from exc


@router.post("/upload_field_data")
def upload_field_data(
    asset_id: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    inspector_name: str = Form(...),
    notes: str = Form(""),
    photo: UploadFile | None = File(default=None),
) -> JSONResponse:
    agent = AssetAppraisalAgent()
    photo_bytes: Optional[bytes] = None
    photo_name: Optional[str] = None

    if photo is not None:
        photo_bytes = photo.file.read()
        photo_name = photo.filename

    try:
        stored = agent.store_field_data(
            asset_id=asset_id,
            inspector_name=inspector_name,
            latitude=latitude,
            longitude=longitude,
            notes=notes,
            photo_bytes=photo_bytes,
            photo_name=photo_name,
        )
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to store field data: {exc}") from exc

    return JSONResponse({"status": "ok", **stored})
