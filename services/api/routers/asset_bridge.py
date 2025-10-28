from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

RUNS_ROOT = Path(__file__).resolve().parents[1] / ".runs" / "asset_bridge"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

router = APIRouter(prefix="/v1/asset-bridge", tags=["asset_bridge"])


def _persist_bridge(df: pd.DataFrame, filename: str | None = None) -> Dict[str, Any]:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    slug = filename or "asset_bridge"
    slug = slug.replace(" ", "_").replace("..", ".")
    bridge_id = f"bridge_{timestamp}_{os.getpid()}"
    dest = RUNS_ROOT / f"{bridge_id}.csv"
    df.to_csv(dest, index=False)

    status_counts: Dict[str, int] = {}
    status_col_candidates = [
        "collateral_status",
        "status",
        "asset_status",
    ]
    for col in status_col_candidates:
        if col in df.columns:
            status_counts = (
                df[col]
                .fillna("unknown")
                .astype(str)
                .str.lower()
                .value_counts()
                .to_dict()
            )
            break

    meta = {
        "bridge_id": bridge_id,
        "path": str(dest),
        "rows": int(df.shape[0]),
        "columns": list(map(str, df.columns)),
        "status_counts": status_counts,
    }
    with open(dest.with_suffix(".json"), "w", encoding="utf-8") as f:
        import json

        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


@router.post("/upload")
async def upload_asset_results(file: UploadFile = File(...)) -> JSONResponse:
    if file.content_type not in {"text/csv", "application/vnd.ms-excel", "application/csv", "text/plain"}:
        # Accept CSV-like mime types but reject obviously wrong ones
        raise HTTPException(status_code=400, detail="Asset appraisal upload must be a CSV file.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {exc}") from exc

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV contained no rows.")

    meta = _persist_bridge(df, filename=file.filename)
    meta["preview"] = df.head(10).to_dict(orient="records")
    return JSONResponse({"status": "ok", **meta})


@router.get("/{bridge_id}")
async def get_bridge_meta(bridge_id: str) -> JSONResponse:
    csv_path = RUNS_ROOT / f"{bridge_id}.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"Bridge '{bridge_id}' not found.")

    meta_path = csv_path.with_suffix(".json")
    meta: Dict[str, Any] = {
        "bridge_id": bridge_id,
        "path": str(csv_path),
        "rows": 0,
        "columns": [],
        "status_counts": {},
    }
    if meta_path.exists():
        import json

        with open(meta_path, "r", encoding="utf-8") as f:
            stored = json.load(f)
        meta.update(stored)

    if meta.get("rows", 0) == 0:
        df = pd.read_csv(csv_path)
        meta["rows"] = int(df.shape[0])
        meta["columns"] = list(map(str, df.columns))
    return JSONResponse({"status": "ok", **meta})
