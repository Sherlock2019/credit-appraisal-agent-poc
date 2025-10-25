# agents/credit_appraisal/agent.py
from __future__ import annotations

import os
import json
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .model_utils import ensure_model

try:  # pragma: no cover - optional dependency during bootstrapping
    from agents.asset_appraisal import AssetAppraisalAgent
except Exception:  # pragma: no cover - fallback when asset agent is absent
    AssetAppraisalAgent = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Paths / Runs root
# ──────────────────────────────────────────────────────────────────────────────

def _discover_runs_root() -> str:
    """
    Find services/api/.runs starting from this file's directory, unless RUNS_ROOT env is set.
    """
    env_root = os.getenv("RUNS_ROOT")
    if env_root:
        os.makedirs(env_root, exist_ok=True)
        return env_root

    here = os.path.abspath(os.path.dirname(__file__))         # agents/credit_appraisal
    proj = os.path.abspath(os.path.join(here, "..", ".."))    # project root (demo-library)
    candidate = os.path.join(proj, "services", "api", ".runs")
    os.makedirs(candidate, exist_ok=True)
    return candidate


RUNS_ROOT = _discover_runs_root()


def _mk_run_dir(run_id: str) -> str:
    d = os.path.join(RUNS_ROOT, run_id)
    os.makedirs(d, exist_ok=True)
    return d


# ──────────────────────────────────────────────────────────────────────────────
# Asset integration helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_asset_context() -> Dict[str, Any] | None:
    if AssetAppraisalAgent is None:
        return None
    try:
        agent = AssetAppraisalAgent()
        return agent.get_latest_asset_record()
    except Exception:
        return None


def _extract_loan_amount(row: pd.Series) -> float:
    for key in (
        "requested_amount",
        "loan_amount",
        "requested_loan_amount",
        "loan_amt",
        "amount_requested",
    ):
        value = row.get(key)
        if value not in (None, ""):
            parsed = _safe_float(value, 0.0)
            if parsed and parsed > 0:
                return float(parsed)
    return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Helpers: rules & proposals
# ──────────────────────────────────────────────────────────────────────────────

def _as_float(d: Dict[str, Any], key: str, default: float) -> float:
    try:
        return float(d.get(key, default))
    except Exception:
        return default


def _as_int(d: Dict[str, Any], key: str, default: int) -> int:
    try:
        return int(float(d.get(key, default)))
    except Exception:
        return default


def _parse_list_of_ints(csv_str: str) -> List[int]:
    if not csv_str:
        return []
    try:
        return [int(x.strip()) for x in str(csv_str).split(",") if x.strip()]
    except Exception:
        return []


def _apply_rules_classic(
    row: pd.Series,
    params: Dict[str, Any],
) -> Tuple[str, Dict[str, bool], Dict[str, Any]]:
    """
    Classic bank-style rules. Returns:
      decision ("approved"/"denied"),
      rule_reasons (dict of metric_name -> True/False),
      proposal (dict with proposed_loan_option or proposed_consolidation_loan)
    """
    max_dti = _as_float(params, "max_debt_to_income", 0.45)
    min_emp_years = _as_int(params, "min_employment_years", 2)
    min_credit_hist = _as_int(params, "min_credit_history_length", 3)
    salary_floor = _as_float(params, "salary_floor", 3000.0)
    max_delinquencies = _as_int(params, "max_num_delinquencies", 2)
    max_current_loans = _as_int(params, "max_current_loans", 3)
    req_min = _as_float(params, "requested_amount_min", 1000.0)
    req_max = _as_float(params, "requested_amount_max", 200000.0)
    allowed_terms = _parse_list_of_ints(params.get("loan_term_months_allowed", ""))

    min_income_debt_ratio = _as_float(params, "min_income_debt_ratio", 0.35)
    compounded_debt_factor = _as_float(params, "compounded_debt_factor", 1.0)
    monthly_debt_relief = _as_float(params, "monthly_debt_relief", 0.50)

    # Pull fields with fallbacks
    dti = float(row.get("debt_to_income", row.get("DTI", 0.0)) or 0.0)

    emp_years_val = row.get("employment_years", row.get("employment_length"))
    emp_years = int(_safe_float(emp_years_val, min_emp_years) or min_emp_years)

    credit_hist_val = row.get("credit_history_length")
    credit_hist = int(_safe_float(credit_hist_val, min_credit_hist) or min_credit_hist)

    income = float(_safe_float(row.get("income"), salary_floor) or salary_floor)
    delinq = int(_safe_float(row.get("num_delinquencies"), 0) or 0)
    current_loans = int(_safe_float(row.get("current_loans"), 0) or 0)

    requested_val = row.get("requested_amount", row.get("loan_amount"))
    requested = float(_safe_float(requested_val, req_min) or req_min)

    term_val = row.get("loan_term_months", row.get("loan_duration_months"))
    default_term = allowed_terms[0] if allowed_terms else 0
    term = int(_safe_float(term_val, default_term) or default_term)

    existing_debt = float(_safe_float(row.get("existing_debt"), 0.0) or 0.0)

    # Debt pressure
    compounded_debt = existing_debt + compounded_debt_factor * requested
    income_debt_ratio = (income / (compounded_debt + 1e-9)) if compounded_debt > 0 else 999.0

    checks = {
        "max_dti": dti <= max_dti,
        "min_emp_years": emp_years >= min_emp_years,
        "min_credit_hist": credit_hist >= min_credit_hist,
        "salary_floor": income >= salary_floor,
        "max_delinquencies": delinq <= max_delinquencies,
        "max_current_loans": current_loans <= max_current_loans,
        "requested_min": requested >= req_min,
        "requested_max": requested <= req_max,
        "allowed_term": (term in allowed_terms) if allowed_terms else True,
        "min_income_debt_ratio": income_debt_ratio >= min_income_debt_ratio,
    }

    approved = all(checks.values())

    proposal: Dict[str, Any] = {}
    if approved:
        proposal["proposed_loan_option"] = {
            "type": "standard",
            "amount": round(requested, 2),
            "term_months": term,
            "monthly_relief_factor": monthly_debt_relief,
        }
        decision = "approved"
    else:
        # If denied and borrower has loans, propose consolidation ("buyback")
        if current_loans > 0 or existing_debt > 0:
            proposal["proposed_consolidation_loan"] = {
                "type": "consolidation",
                "buyback_amount": round(existing_debt, 2),
                "new_term_months": max(24, term or 36),
                "note": "Consolidate existing debt to improve affordability.",
            }
        decision = "denied"

    return decision, checks, proposal


def _apply_rules_ndi(
    row: pd.Series,
    params: Dict[str, Any],
) -> Tuple[str, Dict[str, bool], Dict[str, Any]]:
    """
    NDI-only rules. Approve if both NDI absolute and NDI / income ratio pass thresholds.
    """
    ndi_value = _as_float(params, "ndi_value", 800.0)
    ndi_ratio = _as_float(params, "ndi_ratio", 0.50)

    income = float(row.get("income", 0.0) or 0.0)
    monthly_expenses = float(row.get("monthly_expenses", 0.0) or 0.0)
    monthly_debt_payments = float(row.get("monthly_debt_payments", row.get("existing_debt", 0.0)) or 0.0)

    ndi = income - monthly_expenses - monthly_debt_payments
    ratio = (ndi / (income + 1e-9)) if income > 0 else 0.0

    checks = {
        "ndi_value": ndi >= ndi_value,
        "ndi_ratio": ratio >= ndi_ratio,
    }

    decision = "approved" if all(checks.values()) else "denied"
    proposal: Dict[str, Any] = {}
    if decision == "approved":
        proposal["proposed_loan_option"] = {
            "type": "ndi_approved",
            "note": "NDI thresholds satisfied.",
        }
    else:
        if monthly_debt_payments > 0:
            proposal["proposed_consolidation_loan"] = {
                "type": "consolidation",
                "buyback_amount": round(monthly_debt_payments * 12, 2),
                "note": "Reduce obligations to improve NDI.",
            }

    return decision, checks, proposal


# ──────────────────────────────────────────────────────────────────────────────
# Core pipeline
# ──────────────────────────────────────────────────────────────────────────────

def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _run_core(df_in: pd.DataFrame, **params) -> Dict[str, Any]:
    """
    Core execution:
      - load selected/production/latest model
      - align features robustly
      - compute probabilities
      - apply rules (classic / ndi) gated by score threshold
      - persist artifacts: merged.csv, scores.csv, explanations.csv, summary.json
    Returns JSON-serializable dict (no raw DataFrame to avoid FastAPI serialization errors)
    """
    if df_in is None or df_in.empty:
        raise ValueError("Empty input dataframe.")

    df = df_in.copy()

    # — Model selection from UI/API params
    selected_model_name = params.get("selected_model_name") or None

    # — Load model & feature names (accepts selected_model_name)
    model, feature_cols = ensure_model(df, selected_model_name=selected_model_name)

    # — Feature alignment (order + fill)
    if not isinstance(feature_cols, list):
        feature_cols = list(feature_cols) if feature_cols is not None else []
    if not feature_cols:
        # fall back defensively: numeric-only
        feature_cols = sorted(df.select_dtypes(include=[np.number]).columns.tolist())
        if not feature_cols:
            raise ValueError("No usable numeric features for inference.")

    X = df.reindex(columns=feature_cols, fill_value=0.0).to_numpy(dtype=float)

    # — Predict probabilities (graceful fallback)
    try:
        probs = model.predict_proba(X)[:, 1]
    except Exception:
        preds = model.predict(X)
        probs = (preds.astype(float) + 0.1) / 1.2
    probs = np.clip(probs, 0.0, 1.0)
    df["score"] = probs
    df["base_score"] = df["score"]

    # Asset context integration (collateral influence)
    asset_context = _load_asset_context() or {}
    asset_value = _safe_float(asset_context.get("estimated_value"), 0.0) or 0.0
    asset_confidence = _safe_float(asset_context.get("confidence"), 0.0) or 0.0
    asset_legitimacy = _safe_float(asset_context.get("legitimacy_score"), 0.0) or 0.0
    asset_verified = bool(asset_context.get("verified", False))
    asset_source = asset_context.get("source")
    asset_id = asset_context.get("asset_id")
    asset_type = asset_context.get("asset_type")
    target_ltv = _as_float(params, "target_ltv", 0.8)

    index_list = list(df.index)
    adjustment_factors: List[float] = []
    ltv_values: List[float | None] = []

    if asset_value > 0:
        for idx in index_list:
            row = df.loc[idx]
            base_score = _safe_float(row.get("base_score"), 0.0)
            loan_amount = _extract_loan_amount(row)
            if loan_amount > 0:
                ltv = loan_amount / asset_value
                ltv_values.append(round(ltv, 4))
                if ltv <= target_ltv:
                    coverage_factor = min(1.15, 1.0 + (target_ltv - ltv) * 0.1)
                else:
                    coverage_factor = max(0.8, 1.0 - (ltv - target_ltv) * 0.2)
            else:
                ltv = None
                ltv_values.append(None)
                coverage_factor = 1.0

            if asset_legitimacy:
                legitimacy_delta = max(-0.05, min(0.05, (asset_legitimacy - 0.9) * 0.1))
                legitimacy_factor = 1.0 + legitimacy_delta
            else:
                legitimacy_factor = 1.0

            if asset_confidence:
                confidence_delta = max(-0.05, min(0.05, (asset_confidence - 0.85) * 0.1))
                confidence_factor = 1.0 + confidence_delta
            else:
                confidence_factor = 1.0

            verification_factor = 1.03 if asset_verified else 0.98
            factor = coverage_factor * legitimacy_factor * confidence_factor * verification_factor
            factor = max(0.75, min(factor, 1.2))
            adjustment_factors.append(round(factor, 4))

            adjusted_score = float(np.clip((base_score or 0.0) * factor, 0.0, 1.0))
            df.at[idx, "score"] = adjusted_score
    else:
        ltv_values = [None for _ in index_list]
        adjustment_factors = [1.0 for _ in index_list]

    df["asset_ltv"] = ltv_values
    df["asset_adjustment_factor"] = adjustment_factors

    if asset_value > 0:
        df["asset_value"] = asset_value
        df["asset_confidence"] = asset_confidence
        df["asset_legitimacy_score"] = asset_legitimacy or None
        df["asset_verified"] = asset_verified
        df["asset_source"] = asset_source
        df["asset_id"] = asset_id
        df["asset_type"] = asset_type
    else:
        df["asset_value"] = None
        df["asset_confidence"] = None
        df["asset_legitimacy_score"] = None
        df["asset_verified"] = None
        df["asset_source"] = None
        df["asset_id"] = None
        df["asset_type"] = None

    # — Threshold logic (target rate override supported)
    threshold = params.get("threshold")
    target_rate = params.get("target_approval_rate")
    random_band = str(params.get("random_band", params.get("random_approval_band", "false"))).lower() == "true"

    if (threshold in (None, "", "None")) and target_rate not in (None, "", "None"):
        try:
            target = float(target_rate)
            threshold = float(np.quantile(probs, 1.0 - target))
        except Exception:
            threshold = 0.5
    else:
        try:
            threshold = float(threshold)
        except Exception:
            threshold = float(np.random.uniform(0.2, 0.6)) if random_band else 0.5

    # — Rule mode
    rule_mode = (params.get("rule_mode") or "classic").lower()

    # — Row-wise rules gated by model threshold
    decisions: List[str] = []
    reasons: List[Dict[str, bool]] = []
    proposals: List[Dict[str, Any]] = []
    top_feature = "score"

    for idx in index_list:
        row = df.loc[idx]
        model_pass = _safe_float(row.get("score"), 0.0) >= float(threshold)
        row_reasons = {"model_threshold": model_pass}

        if rule_mode == "ndi":
            dec, checks, prop = _apply_rules_ndi(row, params)
        else:
            dec, checks, prop = _apply_rules_classic(row, params)

        final_decision = dec if model_pass else "denied"
        row_reasons.update(checks)

        decisions.append(final_decision)
        reasons.append(row_reasons)
        proposals.append(prop)

    df["decision"] = decisions
    df["rule_reasons"] = [json.dumps(r, ensure_ascii=False) for r in reasons]
    df["top_feature"] = top_feature

    # Flatten proposal columns
    df["proposed_loan_option"] = [
        json.dumps(p.get("proposed_loan_option")) if p.get("proposed_loan_option") else None
        for p in proposals
    ]
    df["proposed_consolidation_loan"] = [
        json.dumps(p.get("proposed_consolidation_loan")) if p.get("proposed_consolidation_loan") else None
        for p in proposals
    ]

    # Summary
    counts = df["decision"].value_counts().to_dict()
    counts.setdefault("approved", 0)
    counts.setdefault("denied", 0)
    total_count = int(sum(counts.values()))

    summary: Dict[str, Any] = {
        "counts": counts,
        "count": total_count,
        "threshold": float(threshold),
        "rule_mode": rule_mode,
        "n_rows": int(len(df)),
        "currency_code": params.get("currency_code"),
        "currency_symbol": params.get("currency_symbol"),
        "selected_model_name": selected_model_name,
    }
    summary.update({k: int(v) for k, v in counts.items()})

    if asset_value > 0:
        summary["asset_context"] = {
            "asset_id": asset_id,
            "asset_type": asset_type,
            "estimated_value": asset_value,
            "confidence": asset_confidence,
            "legitimacy_score": asset_legitimacy,
            "verified": asset_verified,
            "source": asset_source,
        }

    # Run id + artifacts
    run_id = params.get("run_id") or f"run_{int(time.time())}"
    run_dir = _mk_run_dir(run_id)

    # Primary artifact (merged.csv)
    df.to_csv(os.path.join(run_dir, "merged.csv"), index=False)

    # scores.csv (id, score)
    score_id_col = None
    for guess in ("application_id", "id", "app_id"):
        if guess in df.columns:
            score_id_col = guess
            break
    if score_id_col:
        df[[score_id_col, "score"]].to_csv(os.path.join(run_dir, "scores.csv"), index=False)
    else:
        df[["score"]].to_csv(os.path.join(run_dir, "scores.csv"), index=False)

    # explanations.csv (id, top_feature)
    if score_id_col:
        df[[score_id_col, "top_feature"]].to_csv(os.path.join(run_dir, "explanations.csv"), index=False)
    else:
        df[["top_feature"]].to_csv(os.path.join(run_dir, "explanations.csv"), index=False)

    # summary.json
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    explanation_columns = [
        "application_id",
        "decision",
        "score",
        "rule_reasons",
        "proposed_loan_option",
        "proposed_consolidation_loan",
        "asset_value",
        "asset_ltv",
        "asset_adjustment_factor",
    ]
    available_cols = [col for col in explanation_columns if col in df.columns]
    explanations = df[available_cols].to_dict(orient="records")

    # NOTE: return JSON-serializable only (no DataFrame) to avoid FastAPI serialization errors.
    return {
        "run_id": run_id,
        "summary": summary,
        "explanations": explanations,
        "artifacts": {
            "merged_csv": f"{run_dir}/merged.csv",
            "scores_csv": f"{run_dir}/scores.csv",
            "explanations_csv": f"{run_dir}/explanations.csv",
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Public entrypoints (router will try run() first, then legacy)
# ──────────────────────────────────────────────────────────────────────────────

def run(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preferred entrypoint (router will call this if present):
      run(df, params_dict)
    """
    return _run_core(df, **(params or {}))


def run_credit_appraisal(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """
    Legacy entrypoint (router fallback):
      run_credit_appraisal(df, **kwargs)
    """
    return _run_core(df, **kwargs)


# Optional compatibility aliases
def execute(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    return _run_core(df, **kwargs)


def main(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    return _run_core(df, **kwargs)


def run_agent(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    return _run_core(df, **kwargs)
