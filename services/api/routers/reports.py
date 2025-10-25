# services/api/routers/reports.py
from __future__ import annotations

import json
from pathlib import Path as FilePath

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fpdf import FPDF

from services.api.routers.runs import _find_run_dir

router = APIRouter()


def _ensure_pdf(run_dir: FilePath, run_id: str, summary: dict) -> FilePath:
    pdf_path = run_dir / f"{run_id}_credit_report.pdf"
    if pdf_path.exists():
        return pdf_path

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(0, 10, txt="Credit Appraisal Report", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, txt=f"Run ID: {run_id}", ln=True)
    pdf.ln(4)

    if summary:
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 8, txt="Summary", ln=True)
        pdf.set_font("Arial", size=10)
        for key, value in summary.items():
            if isinstance(value, (dict, list)):
                pdf.multi_cell(0, 6, txt=f"- {key}: {json.dumps(value, ensure_ascii=False)[:200]}")
            else:
                pdf.cell(0, 6, txt=f"- {key}: {value}", ln=True)
    else:
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 6, txt="No summary metadata available.", ln=True)

    pdf.output(str(pdf_path))
    return pdf_path


@router.get("/v1/runs/{run_id}/report")
def get_report(run_id: str, format: str = "csv"):
    run_dir = _find_run_dir(run_id)

    summary_path = run_dir / "summary.json"
    summary = {}
    if summary_path.exists():
        with summary_path.open() as fh:
            summary = json.load(fh)

    if format == "csv":
        merged_path = run_dir / "merged.csv"
        if not merged_path.exists():
            raise HTTPException(404, detail="merged.csv not found")
        return FileResponse(str(merged_path), media_type="text/csv", filename=f"merged_{run_id}.csv")

    if format == "scores_csv":
        scores_path = run_dir / "scores.csv"
        if not scores_path.exists():
            raise HTTPException(404, detail="scores.csv not found")
        return FileResponse(str(scores_path), media_type="text/csv", filename=f"scores_{run_id}.csv")

    if format == "explanations_csv":
        expl_path = run_dir / "explanations.csv"
        if not expl_path.exists():
            raise HTTPException(404, detail="explanations.csv not found")
        return FileResponse(str(expl_path), media_type="text/csv", filename=f"explanations_{run_id}.csv")

    if format == "json":
        bundle = {
            "run_id": run_id,
            "summary": summary,
        }
        scores_json = run_dir / "scores.json"
        expl_json = run_dir / "explanations.json"
        if scores_json.exists():
            with scores_json.open() as fh:
                bundle["scores"] = json.load(fh)
        if expl_json.exists():
            with expl_json.open() as fh:
                bundle["explanations"] = json.load(fh)
        return JSONResponse(bundle)

    if format == "pdf":
        pdf_path = _ensure_pdf(run_dir, run_id, summary)
        return FileResponse(str(pdf_path), media_type="application/pdf", filename=pdf_path.name)

    raise HTTPException(400, detail="unknown format")

