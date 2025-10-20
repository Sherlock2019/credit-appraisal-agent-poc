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
    pdf.cell(0, 8, f"Threshold used: {summary.get('threshold_used')}", ln=True)
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
            if len(m): return int(m[0])
    if 1 in classes:
        return int(np.where(classes == 1)[0][0])
    return -1

def _pmt(principal: float, apr: float, months: int) -> float:
    """Amortized monthly payment. apr = e.g., 0.12 for 12%."""
    if months <= 0: return float("nan")
    r = apr / 12.0
    if r == 0: return principal / months
    return principal * (r / (1 - (1 + r) ** (-months)))

@agent.runner
def run(inputs: dict, ctx: dict):
    """
    inputs: { "applications_csv": path }
    ctx.tuning:
        - threshold: float (manual)
        - target_approval_rate: float (0..1)
        - random_band: bool → if True and threshold/target not set, pick 20–60% randomly
        - loan_term_months_allowed: list[int]
        - min_income_debt_ratio, compounded_debt_factor, monthly_debt_relief, salary_floor (for proposals)
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

    # -------- Tuning --------
    thr = 0.45
    target_rate = None
    narrative = None
    rand_band = False
    tuning = {}
    if isinstance(ctx, dict):
        narrative = ctx.get("narrative")
        tuning = ctx.get("tuning") or {}
        if tuning.get("threshold") not in (None, "", "None", "null"):
            thr = float(tuning["threshold"])
        if tuning.get("target_approval_rate") not in (None, "", "None", "null"):
            target_rate = float(tuning["target_approval_rate"])
        rand_band = bool(tuning.get("random_band", False))

    desired = None
    if target_rate is not None and 0.0 < target_rate < 1.0 and len(proba) > 0:
        thr = float(np.quantile(proba, 1.0 - target_rate))
    elif "threshold" not in tuning and rand_band and len(proba) > 0:
        desired = float(np.random.uniform(0.20, 0.60))
        thr = float(np.quantile(proba, 1.0 - desired))

    print(f"[DEBUG] credit_appraisal: random_band={rand_band}, target_rate={target_rate}, "
          f"chosen_random={desired}, final_threshold={thr:.3f}")

    decision = np.where(proba >= thr, "approve", "deny")

    # knobs for proposals
    allowed_terms = tuning.get("loan_term_months_allowed") or [36, 48, 60]
    if isinstance(allowed_terms, (int, float, str)):
        allowed_terms = [int(float(allowed_terms))]
    allowed_terms = [int(t) for t in allowed_terms] or [36]

    min_ratio = float(tuning.get("min_income_debt_ratio") or 0.35)
    factor = float(tuning.get("compounded_debt_factor") or 1.0)
    relief = float(tuning.get("monthly_debt_relief") or 0.0)
    salary_floor = float(tuning.get("salary_floor") or 0.0)

    # build outputs
    items: List[Dict] = []
    for i in range(len(df)):
        row = df.iloc[i]
        sv = shap_values[i]
        idx = int(np.argmax(np.abs(sv)))
        feature = FEATURES[idx]
        direction = "increased" if sv[idx] > 0 else "decreased"
        explanation = f"{feature} {direction} approval likelihood most"

        # inputs for proposals
        income = float(row.get("income", 0.0))
        req = float(row.get("requested_amount", row.get("loan_amount", 0.0)))
        term = int(row.get("loan_term_months", 36)) or 36
        term = term if term in allowed_terms else allowed_terms[0]
        exist_debt = float(row.get("existing_debt", 0.0))
        monthly_income = max(salary_floor, income / 12.0)

        # approval → propose a loan option
        proposed_option = None
        if decision[i] == "approve":
            # rate tied to risk (lower score -> higher rate): 6–18%
            apr = 0.06 + (1.0 - float(proba[i])) * 0.12
            payment = _pmt(req, apr, term)
            proposed_option = {
                "term_months": int(term),
                "apr": round(apr * 100, 2),
                "est_monthly_payment": round(payment, 2),
            }

        # denial → propose a consolidation/buy-back loan
        proposed_consol = None
        if decision[i] == "deny":
            # target max monthly debt allowed by min_ratio
            lt = term if term >= 12 else 36
            exist_monthly = exist_debt / lt
            req_monthly = req / lt
            compounded = (exist_monthly + factor * req_monthly) * (1.0 - relief)
            # allowable monthly debt to pass min_ratio
            allowed_monthly_debt = monthly_income / max(min_ratio, 1e-6)
            target_payment = max(0.0, allowed_monthly_debt)  # total monthly across debts
            # propose a single consolidation loan that replaces existing+requested
            target_months = max(allowed_terms) if allowed_terms else 60
            apr_guess = 0.10 + (1.0 - float(proba[i])) * 0.10  # 10–20%
            # invert payment formula roughly: P ≈ payment * annuity factor
            r = apr_guess / 12.0
            if r == 0:
                principal = target_payment * target_months
            else:
                ann = (1 - (1 + r) ** (-target_months)) / r
                principal = target_payment * ann
            principal = max(principal, 0.0)
            proposed_consol = {
                "term_months": int(target_months),
                "apr": round(apr_guess * 100, 2),
                "max_monthly_payment": round(target_payment, 2),
                "estimated_principal": round(principal, 2),
                "note": "Single new loan to buy back existing + new request to meet policy ratio."
            }

        items.append({
            "application_id": str(row.get("application_id", f"row_{i}")),
            "score": float(proba[i]),
            "decision": str(decision[i]),
            "top_feature": feature,
            "shap": float(sv[idx]),
            "explanation": explanation,
            "proposed_loan_option": proposed_option,
            "proposed_consolidation_loan": proposed_consol,
        })

    approved = int((decision == "approve").sum())
    summary = {
        "count": int(len(df)),
        "approved": approved,
        "denied": int(len(df) - approved),
        "threshold_used": float(thr),
        "target_rate_used": float(target_rate) if target_rate is not None else None,
    }

    run_id = f"run_{uuid.uuid4().hex}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    scores_df = pd.DataFrame([{"application_id": it["application_id"], "score": it["score"], "decision": it["decision"]} for it in items])
    explanations_df = pd.DataFrame(items)

    scores_csv_text = scores_df.to_csv(index=False)
    explanations_csv_text = explanations_df.to_csv(index=False)

    (run_dir / "scores.csv").write_text(scores_csv_text, encoding="utf-8")
    (run_dir / "explanations.csv").write_text(explanations_csv_text, encoding="utf-8")
    (run_dir / "scores.json").write_text(json.dumps(scores_df.to_dict(orient="records")), encoding="utf-8")
    (run_dir / "df.json").write_text(json.dumps(explanations_df.to_dict(orient="records")), encoding="utf-8")
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
