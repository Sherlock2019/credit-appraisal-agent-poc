# services/api/routers/runs.py
from __future__ import annotations

import os
import json
from pathlib import Path as FilePath
from typing import Literal

from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

ROOT = FilePath(__file__).resolve().parents[3]
RUNS_ROOT = ROOT / "services" / "api" / ".runs"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

router = APIRouter(prefix="/v1/runs", tags=["runs"])


def _find_run_dir(run_id: str) -> FilePath:
    d = RUNS_ROOT / run_id
    if d.is_dir():
        return d
    for cand in RUNS_ROOT.iterdir():
        if cand.is_dir() and cand.name == run_id:
            return cand
    raise HTTPException(status_code=404, detail="run not found")


@router.get("/{run_id}")
def get_run(run_id: str = Path(..., description="Run ID returned from /agents/run")):
    run_dir = _find_run_dir(run_id)
    summary_json = run_dir / "summary.json"
    if summary_json.exists():
        with summary_json.open() as f:
            summary = json.load(f)
    else:
        summary = {}
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "artifacts": {
            "scores_csv": str(run_dir / "scores.csv"),
            "explanations_csv": str(run_dir / "explanations.csv"),
            "merged_csv": str(run_dir / "merged.csv"),
            "scores_json": str(run_dir / "scores.json"),
            "explanations_json": str(run_dir / "explanations.json"),
            "summary_json": str(summary_json),
            "pdf_report": str(run_dir / f"{run_id}_credit_report.pdf"),
        },
        "summary": summary,
    }


@router.get("/{run_id}/report")
def download_report(
    run_id: str = Path(..., description="Run ID"),
    format: Literal["pdf", "scores_csv", "explanations_csv", "csv", "json"] = Query(
        "pdf",
        description="Pick one: pdf (report), scores_csv, explanations_csv, csv (merged), json (combined JSON).",
    ),
):
    run_dir = _find_run_dir(run_id)

    if format == "pdf":
        path = run_dir / f"{run_id}_credit_report.pdf"
        if not path.exists():
            raise HTTPException(404, "PDF not found")
        return FileResponse(str(path), media_type="application/pdf", filename=path.name)

    if format == "scores_csv":
        path = run_dir / "scores.csv"
        if not path.exists():
            raise HTTPException(404, "Scores CSV not found")
        return FileResponse(str(path), media_type="text/csv", filename=f"scores_{run_id}.csv")

    if format == "explanations_csv":
        path = run_dir / "explanations.csv"
        if not path.exists():
            raise HTTPException(404, "Explanations CSV not found")
        return FileResponse(str(path), media_type="text/csv", filename=f"explanations_{run_id}.csv")

    if format == "csv":
        path = run_dir / "merged.csv"
        if not path.exists():
            raise HTTPException(404, "Merged CSV not found")
        return FileResponse(str(path), media_type="text/csv", filename=f"merged_{run_id}.csv")

    if format == "json":
        # Build a compact JSON bundle
        scores_json = run_dir / "scores.json"
        expl_json = run_dir / "explanations.json"
        summary_json = run_dir / "summary.json"

        data = {}
        if scores_json.exists():
            with scores_json.open() as fh:
                data["scores"] = json.load(fh)
        if expl_json.exists():
            with expl_json.open() as fh:
                data["explanations"] = json.load(fh)
        if summary_json.exists():
            with summary_json.open() as fh:
                data["summary"] = json.load(fh)
        return JSONResponse(data)

    return PlainTextResponse("Unsupported format", status_code=400)
