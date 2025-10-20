from __future__ import annotations

import io
import os
import re
import json
import uuid
import shutil
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from agents.credit_appraisal.agent import agent as credit_agent  # type: ignore

router = APIRouter(prefix="/v1/agents", tags=["agents"])

ROOT = os.path.expanduser("~/demo-library")
RUNS_ROOT = os.path.join(ROOT, "services", "api", ".runs")
os.makedirs(RUNS_ROOT, exist_ok=True)

SAMPLE_CSV = os.path.join(ROOT, "agents", "credit_appraisal", "sample_data", "credit_sample.csv")

def _boolish(v, default=False) -> bool:
    if isinstance(v, bool): return v
    if v is None: return default
    s = str(v).strip().lower()
    if s in {"true","1","yes","y"}: return True
    if s in {"false","0","no","n"}: return False
    return default

def parse_terms_csv(s: Optional[str]) -> List[int]:
    if not s: return []
    out: List[int] = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok: continue
        try: out.append(int(float(tok)))
        except Exception: pass
    return out

_num_pat = re.compile(r"[-+]?\d*\.?\d+")
def coerce_numeric(series: pd.Series, as_int: bool = False) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series): return series
    s = series.astype(str).str.replace(",", "", regex=False)
    s = s.str.extract(_num_pat, expand=False)
    vals = pd.to_numeric(s, errors="coerce")
    return vals.round().astype("Int64") if as_int else vals

def normalize_agent_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "employment_years" not in out.columns and "employment_length" in out.columns:
        out["employment_years"] = out["employment_length"]
    if "loan_term_months" not in out.columns and "loan_duration_months" in out.columns:
        out["loan_term_months"] = out["loan_duration_months"]
    if "debt_to_income" not in out.columns and "DTI" in out.columns:
        out["debt_to_income"] = out["DTI"]
    return out

def apply_tuning_filters(df: pd.DataFrame, filt: Dict[str, Any]) -> pd.DataFrame:
    """Policy filters + income/compounded-debt ratio."""
    out = normalize_agent_schema(df)

    # coerce basics
    for name in ["employment_years","credit_history_length","num_delinquencies",
                 "current_loans","loan_term_months"]:
        if name in out.columns:
            out[name] = coerce_numeric(out[name], as_int=True)
    for name in ["debt_to_income","requested_amount","income","existing_debt"]:
        if name in out.columns:
            out[name] = coerce_numeric(out[name]).astype(float)

    # Basic caps/thresholds
    m = pd.Series(True, index=out.index, dtype=bool)
    v = filt.get("min_employment_years")
    if v is not None and "employment_years" in out.columns:
        m &= out["employment_years"].fillna(-10) >= int(v)

    v = filt.get("max_debt_to_income")
    if v is not None and "debt_to_income" in out.columns:
        col = out["debt_to_income"].copy()
        # normalize percent-style DTI if needed
        if col.dropna().gt(1.2).any(): col = col / 100.0
        m &= col.fillna(np.inf) <= float(v)

    v = filt.get("min_credit_history_length")
    if v is not None and "credit_history_length" in out.columns:
        m &= out["credit_history_length"].fillna(-10) >= int(v)

    v = filt.get("max_num_delinquencies")
    if v is not None and "num_delinquencies" in out.columns:
        m &= out["num_delinquencies"].fillna(10_000) <= int(v)

    v = filt.get("max_current_loans")
    if v is not None and "current_loans" in out.columns:
        m &= out["current_loans"].fillna(10_000) <= int(v)

    vmin = filt.get("requested_amount_min")
    vmax = filt.get("requested_amount_max")
    if "requested_amount" in out.columns:
        if vmin is not None: m &= out["requested_amount"].fillna(-np.inf) >= float(vmin)
        if vmax is not None: m &= out["requested_amount"].fillna(np.inf) <= float(vmax)

    # Allowed terms
    allow = filt.get("loan_term_months_allowed") or []
    if allow and "loan_term_months" in out.columns:
        allowed = set(int(x) for x in allow)
        m &= out["loan_term_months"].isin(list(allowed))

    # -------- income / compounded debt ratio --------
    min_ratio = filt.get("min_income_debt_ratio")
    factor = float(filt.get("compounded_debt_factor") or 1.0)
    relief = float(filt.get("monthly_debt_relief") or 0.0)
    salary_floor = float(filt.get("salary_floor") or 0.0)

    if min_ratio is not None:
        # best-effort monthly normalization
        lt = out["loan_term_months"] if "loan_term_months" in out.columns else pd.Series(36, index=out.index)
        lt = lt.fillna(36).clip(lower=12).astype(float)

        income_monthly = (out["income"].fillna(0.0) / 12.0) if "income" in out.columns else pd.Series(0.0, index=out.index)
        if salary_floor > 0:
            income_monthly = np.maximum(income_monthly, salary_floor)

        existing_monthly = (out["existing_debt"].fillna(0.0) / lt) if "existing_debt" in out.columns else pd.Series(0.0, index=out.index)
        req_monthly = (out["requested_amount"].fillna(0.0) / lt) if "requested_amount" in out.columns else pd.Series(0.0, index=out.index)

        compounded_monthly = existing_monthly + factor * req_monthly
        adjusted_monthly = compounded_monthly * (1.0 - relief)

        comp_ratio = income_monthly / np.maximum(adjusted_monthly, 1.0)
        out["__income_debt_ratio"] = comp_ratio  # keep for debugging
        m &= comp_ratio.fillna(0.0) >= float(min_ratio)

    return out.loc[m].reset_index(drop=True)


@router.post("/credit_appraisal/run")
async def run_agent(
    use_sample: str = Form(default="false"),
    use_llm_narrative: str = Form(default="false"),
    file: Optional[UploadFile] = File(default=None),

    threshold: Optional[float] = Form(default=None),
    target_approval_rate: Optional[float] = Form(default=None),
    random_band: Optional[bool] = Form(default=False),
    random_approval_band: Optional[bool] = Form(default=False),

    min_employment_years: Optional[int] = Form(default=None),
    max_debt_to_income: Optional[float] = Form(default=None),
    min_credit_history_length: Optional[int] = Form(default=None),
    max_num_delinquencies: Optional[int] = Form(default=None),
    max_current_loans: Optional[int] = Form(default=None),
    requested_amount_min: Optional[float] = Form(default=None),
    requested_amount_max: Optional[float] = Form(default=None),
    loan_term_months_allowed: Optional[str] = Form(default=None),

    # new debt-pressure controls
    min_income_debt_ratio: Optional[float] = Form(default=None),
    compounded_debt_factor: Optional[float] = Form(default=None),
    monthly_debt_relief: Optional[float] = Form(default=None),
    salary_floor: Optional[float] = Form(default=None),
):
    try:
        run_id = f"run_{uuid.uuid4().hex}"
        run_dir = os.path.join(RUNS_ROOT, run_id)
        os.makedirs(run_dir, exist_ok=True)

        use_sample_flag = _boolish(use_sample, False)
        if use_sample_flag:
            if not os.path.exists(SAMPLE_CSV):
                raise HTTPException(status_code=500, detail=f"Sample CSV not found at {SAMPLE_CSV}")
            df = pd.read_csv(SAMPLE_CSV)
        else:
            if file is None:
                raise HTTPException(status_code=400, detail="CSV file is required when use_sample=false")
            raw = await file.read()
            if not raw:
                file.file.seek(0)
                raw = await file.read()
            df = pd.read_csv(io.BytesIO(raw))

        filters = {
            "min_employment_years": min_employment_years,
            "max_debt_to_income": max_debt_to_income,
            "min_credit_history_length": min_credit_history_length,
            "max_num_delinquencies": max_num_delinquencies,
            "max_current_loans": max_current_loans,
            "requested_amount_min": requested_amount_min,
            "requested_amount_max": requested_amount_max,
            "loan_term_months_allowed": parse_terms_csv(loan_term_months_allowed),
            # new debt-pressure controls
            "min_income_debt_ratio": min_income_debt_ratio,
            "compounded_debt_factor": compounded_debt_factor,
            "monthly_debt_relief": monthly_debt_relief,
            "salary_floor": salary_floor,
        }
        df_filtered = apply_tuning_filters(df, filters)
        if len(df_filtered) == 0:
            raise HTTPException(status_code=400, detail="All rows were filtered out by the tuning constraints.")

        csv_path = os.path.join(run_dir, "applications.csv")
        df_filtered.to_csv(csv_path, index=False)

        ctx = {
            "narrative": None if (not _boolish(use_llm_narrative, False)) else "Please summarize key portfolio drivers.",
            "tuning": {
                "threshold": threshold,
                "target_approval_rate": target_approval_rate,
                # accept either alias
                "random_band": _boolish(random_band, _boolish(random_approval_band, False)),
                **filters,
            },
        }

        result = credit_agent.run({"applications_csv": csv_path}, ctx)

        scores = result.get("scores", [])
        exps = result.get("explanations", [])
        summary = result.get("summary", {})

        scores_csv = os.path.join(run_dir, "scores.csv")
        exp_csv = os.path.join(run_dir, "explanations.csv")
        merged_csv = os.path.join(run_dir, "merged.csv")
        scores_json = os.path.join(run_dir, "scores.json")
        df_json = os.path.join(run_dir, "df.json")
        summary_json = os.path.join(run_dir, "summary.json")

        pd.DataFrame(scores).to_csv(scores_csv, index=False)
        pd.DataFrame(exps).to_csv(exp_csv, index=False)
        (pd.DataFrame(exps) if exps else pd.DataFrame(scores)).to_csv(merged_csv, index=False)

        with open(scores_json, "w") as f: json.dump(scores, f)
        with open(df_json, "w") as f: json.dump(exps, f)
        with open(summary_json, "w") as f: json.dump(summary, f)

        pdf_out = None
        artifacts = result.get("artifacts", {}) or {}
        if artifacts.get("explanation_pdf") and os.path.exists(artifacts["explanation_pdf"]):
            pdf_out = os.path.join(run_dir, f"{run_id}_credit_report.pdf")
            if os.path.abspath(artifacts["explanation_pdf"]) != os.path.abspath(pdf_out):
                shutil.copyfile(artifacts["explanation_pdf"], pdf_out)

        return JSONResponse({
            "run_id": run_id,
            "result": {
                "run_id": run_id,
                "scores": scores,
                "explanations": exps,
                "summary": summary,
                "scores_csv_text": pd.DataFrame(scores).to_csv(index=False),
                "explanations_csv_text": pd.DataFrame(exps).to_csv(index=False),
                "artifacts": {
                    "run_dir": run_dir,
                    "scores_csv": scores_csv,
                    "explanations_csv": exp_csv,
                    "scores_json": scores_json,
                    "df_json": df_json,
                    "summary_json": summary_json,
                    "explanation_pdf": pdf_out,
                },
            },
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Run error: {type(e).__name__}: {e}")
