# ~/demo-library/agents/credit_appraisal/agent.py
from __future__ import annotations

import os
import uuid
import json
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import shap
from fpdf import FPDF

# 🔧 use package import that matches your tree
from agents.credit_appraisal.model_utils import ensure_model, FEATURES
from agent_platform.agent_sdk import Agent

agent = Agent(name="credit_appraisal", root=os.path.dirname(__file__))

RUNS_DIR = Path.home() / "demo-library" / "services" / "api" / ".runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def make_pdf(report_path: str, items: List[Dict], summary: Dict, narrative: Optional[str]) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Credit Appraisal Report", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Total applications: {summary['count']}", ln=True)
    pdf.cell(0, 8, f"Approved: {summary['approved']} | Denied: {summary['denied']}", ln=True)
    if narrative:
        pdf.ln(4)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Narrative:", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 6, narrative)
    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Top Explanations:", ln=True)
    pdf.set_font("Arial", "", 11)
    for it in items[:10]:
        pdf.multi_cell(
            0, 6,
            f"{it['application_id']}: decision={it['decision']}, "
            f"score={it['score']:.3f}, reason={it['explanation']}"
        )
    pdf.output(report_path)


def _pos_idx_for_approve(classes) -> int:
    if classes is None:
        return 1
    if isinstance(classes[0], str):
        for wanted in ("approve", "approved", "good", "nondefault", "positive"):
            m = np.where(classes == wanted)[0]
            if len(m):
                return int(m[0])
    if 1 in classes:
        return int(np.where(classes == 1)[0][0])
    return -1


@agent.runner
def run(inputs: dict, ctx: dict):
    """
    inputs: { "applications_csv": path }
    ctx:
      - threshold: float (manual)  [default 0.45]
      - target_approval_rate: float in (0,1) to auto-quantile threshold (takes precedence)
      - narrative: optional str for PDF
    """
    csv_path = inputs["applications_csv"]
    df = pd.read_csv(csv_path)

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    model = ensure_model(df)
    X = df[FEATURES]

    classes = getattr(model, "classes_", None)
    pos_idx = _pos_idx_for_approve(classes)
    proba = model.predict_proba(X)[:, pos_idx]

    # SHAP explanations
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[pos_idx]

    # Tuning
    thr = 0.45  # realistic default
    target_rate = None
    narrative = None
    if isinstance(ctx, dict):
        if "threshold" in ctx and str(ctx["threshold"]).strip() not in ("", "None", "null"):
            thr = float(ctx["threshold"])
        if "target_approval_rate" in ctx and str(ctx["target_approval_rate"]).strip() not in ("", "None", "null"):
            target_rate = float(ctx["target_approval_rate"])
        narrative = ctx.get("narrative")

    if target_rate is not None and 0.0 < target_rate < 1.0 and len(proba) > 0:
        thr = float(np.quantile(proba, 1.0 - target_rate))

    decision = np.where(proba >= thr, "approve", "deny")

    # Per-row items
    items: List[Dict] = []
    for i in range(len(df)):
        sv = shap_values[i]
        idx = int(np.argmax(np.abs(sv)))
        feature = FEATURES[idx]
        direction = "increased" if sv[idx] > 0 else "decreased"
        explanation = f"{feature} {direction} approval likelihood most"
        items.append({
            "application_id": str(df.iloc[i].get("application_id", f"row_{i}")),
            "score": float(proba[i]),
            "decision": str(decision[i]),
            "top_feature": feature,
            "shap": float(sv[idx]),
            "explanation": explanation
        })

    approved = int((decision == "approve").sum())
    summary = {
        "count": int(len(df)),
        "approved": approved,
        "denied": int(len(df) - approved),
        "threshold_used": float(thr),
    }

    run_id = f"run_{uuid.uuid4().hex}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    scores_df = pd.DataFrame(
        [{"application_id": it["application_id"], "score": it["score"], "decision": it["decision"]} for it in items]
    )
    explanations_df = pd.DataFrame(items)

    scores_csv_text = scores_df.to_csv(index=False)
    explanations_csv_text = explanations_df.to_csv(index=False)

    (run_dir / "scores.csv").write_text(scores_csv_text, encoding="utf-8")
    (run_dir / "explanations.csv").write_text(explanations_csv_text, encoding="utf-8")
    (run_dir / "scores.json").write_text(json.dumps(scores_df.to_dict(orient="records")), encoding="utf-8")
    (run_dir / "df.json").write_text(json.dumps(scores_df.to_dict(orient="records")), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    report_pdf_path = run_dir / f"{run_id}_credit_report.pdf"
    make_pdf(str(report_pdf_path), items, summary, narrative)

    return {
        "run_id": run_id,
        "scores": scores_df.to_dict(orient="records"),
        "explanations": explanations_df.to_dict(orient="records"),
        "summary": summary,
        "scores_csv_text": scores_csv_text,
        "explanations_csv_text": explanations_csv_text,
        "artifacts": {
            "run_dir": str(run_dir),
            "scores_csv": str(run_dir / "scores.csv"),
            "explanations_csv": str(run_dir / "explanations.csv"),
            "scores_json": str(run_dir / "scores.json"),
            "df_json": str(run_dir / "df.json"),
            "summary_json": str(run_dir / "summary.json"),
            "explanation_pdf": str(report_pdf_path),
        },
    }
