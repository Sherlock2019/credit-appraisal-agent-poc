# ~/demo-library/services/api/routers/reports.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse

ROOT = Path.home() / "demo-library"
RUNS_DIR = ROOT / "services" / "api" / ".runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter(prefix="/v1/runs", tags=["reports"])


@router.get("/{run_id}/report")
def get_report(
    run_id: str,
    format: Literal["pdf", "scores_csv", "explanations_csv", "csv", "json"] = Query("json")
):
    rdir = RUNS_DIR / run_id
    if not rdir.exists():
        raise HTTPException(status_code=404, detail="run_id not found")

    if format == "pdf":
        pdfs = list(rdir.glob("*.pdf"))
        if not pdfs:
            raise HTTPException(status_code=404, detail="PDF not found")
        return FileResponse(str(pdfs[0]), media_type="application/pdf", filename=pdfs[0].name)

    elif format == "scores_csv":
        f = rdir / "scores.csv"
        if not f.exists():
            raise HTTPException(status_code=404, detail="scores_csv file not found")
        return FileResponse(str(f), media_type="text/csv", filename=f.name)

    elif format == "explanations_csv":
        f = rdir / "explanations.csv"
        if not f.exists():
            raise HTTPException(status_code=404, detail="explanations_csv file not found")
        return FileResponse(str(f), media_type="text/csv", filename=f.name)

    elif format == "csv":
        f_scores = rdir / "scores.csv"
        f_expl = rdir / "explanations.csv"
        if not (f_scores.exists() and f_expl.exists()):
            raise HTTPException(status_code=404, detail="scores/explanations CSV not found")
        s = pd.read_csv(f_scores)
        e = pd.read_csv(f_expl)
        keep = ["application_id", "top_feature", "shap", "explanation"]
        merged = s.merge(e[keep], on="application_id", how="left")
        csv_bytes = merged.to_csv(index=False).encode("utf-8")
        return StreamingResponse(iter([csv_bytes]), media_type="text/csv",
                                 headers={"Content-Disposition": f'attachment; filename="merged_{run_id}.csv"'})

    elif format == "json":
        f_summary = rdir / "summary.json"
        f_scores = rdir / "scores.json"
        payload = {}
        if f_summary.exists():
            payload["summary"] = json.loads(f_summary.read_text(encoding="utf-8"))
        if f_scores.exists():
            payload["scores"] = json.loads(f_scores.read_text(encoding="utf-8"))
        return JSONResponse(payload)

    raise HTTPException(status_code=400, detail="Unknown format")
