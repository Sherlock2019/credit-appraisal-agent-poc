# agents/credit_appraisal/agent.py
from __future__ import annotations

import os
import json
import time
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .model_utils import ensure_model


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
# Asset bridge helpers (ingest CSV exported by external asset agent)
# ──────────────────────────────────────────────────────────────────────────────


def _asset_bridge_root() -> Path:
    root = Path(RUNS_ROOT) / "asset_bridge"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _normalize_asset_frame(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [
        _normalize_key(col) or f"col_{idx}"
        for idx, col in enumerate(df.columns)
    ]
    return normalized


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "t", "validated"}:
        return True
    if text in {"false", "0", "no", "n", "f"}:
        return False
    return default


def _row_lookup_key(row: pd.Series, join_key: str) -> str | None:
    if join_key in row.index:
        value = row.get(join_key)
    else:
        lookup = join_key.lower()
        value = None
        for candidate in row.index:
            if str(candidate).lower() == lookup:
                value = row.get(candidate)
                break
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    return text or None


def _load_asset_bridge_table(bridge_id: str | None) -> pd.DataFrame | None:
    if not bridge_id:
        return None
    csv_path = _asset_bridge_root() / f"{bridge_id}.csv"
    if not csv_path.exists():
        return None
    try:
        return pd.read_csv(csv_path)
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

    # Asset bridge integration (collateral influence per application)
    asset_bridge_id = params.get("asset_bridge_id") or params.get("asset_bridge")
    asset_join_key = params.get("asset_join_key") or "application_id"
    target_ltv = _as_float(params, "target_ltv", 0.8)

    raw_asset_df = _load_asset_bridge_table(asset_bridge_id)
    normalized_asset_df: pd.DataFrame | None = None
    asset_lookup: Dict[str, Dict[str, Any]] = {}
    asset_summary: Dict[str, Any] = {}

    if raw_asset_df is not None and not raw_asset_df.empty:
        normalized_asset_df = _normalize_asset_frame(raw_asset_df)
        normalized_join_key = _normalize_key(asset_join_key)

        if normalized_join_key not in normalized_asset_df.columns:
            for col in normalized_asset_df.columns:
                if col.replace("_", "") == normalized_join_key.replace("_", ""):
                    normalized_join_key = col
                    break

        if normalized_join_key in normalized_asset_df.columns:
            asset_frame = normalized_asset_df.dropna(subset=[normalized_join_key]).copy()
            asset_frame[normalized_join_key] = asset_frame[normalized_join_key].astype(str)
            asset_lookup = asset_frame.set_index(normalized_join_key).to_dict(orient="index")
            asset_summary = {
                "bridge_id": asset_bridge_id,
                "rows": int(asset_frame.shape[0]),
                "join_key": asset_join_key,
            }
            if "collateral_status" in asset_frame.columns:
                asset_summary["status_counts"] = (
                    asset_frame["collateral_status"]
                    .fillna("unknown")
                    .astype(str)
                    .str.lower()
                    .value_counts()
                    .to_dict()
                )
            if "include_in_credit" in asset_frame.columns:
                asset_summary["include_counts"] = (
                    asset_frame["include_in_credit"]
                    .fillna(False)
                    .apply(_safe_bool)
                    .value_counts()
                    .to_dict()
                )
        else:
            asset_summary = {
                "bridge_id": asset_bridge_id,
                "warning": f"Join key '{asset_join_key}' not present in asset bridge export.",
            }
    elif asset_bridge_id:
        asset_summary = {
            "bridge_id": asset_bridge_id,
            "warning": "Asset bridge export not found or empty.",
        }

    index_list = list(df.index)
    ltv_values: List[float | None] = []
    adjustment_factors: List[float] = []
    asset_values: List[float | None] = []
    asset_confidences: List[float | None] = []
    asset_legitimacies: List[float | None] = []
    asset_statuses: List[str | None] = []
    asset_stages: List[str | None] = []
    asset_includes: List[bool | None] = []
    asset_overrides: List[str | None] = []
    asset_notes: List[str | None] = []
    asset_updated: List[str | None] = []

    for idx in index_list:
        row = df.loc[idx]
        base_score = float(_safe_float(row.get("base_score"), 0.0) or 0.0)
        loan_amount = _extract_loan_amount(row)

        asset_row = None
        if asset_lookup:
            lookup_key = _row_lookup_key(row, asset_join_key)
            if lookup_key is not None:
                asset_row = asset_lookup.get(lookup_key)

        ltv_value: float | None = None
        factor = 1.0
        asset_value = None
        asset_confidence = None
        asset_legitimacy = None
        asset_status = None
        asset_stage = None
        asset_include: bool | None = None
        asset_override: str | None = None
        asset_note = None
        asset_update = None

        if asset_row:
            asset_value = _safe_float(asset_row.get("collateral_value"), 0.0) or None
            asset_confidence = _safe_float(asset_row.get("confidence"), 0.0) or None
            asset_legitimacy = _safe_float(asset_row.get("legitimacy_score"), 0.0) or None
            asset_status = str(asset_row.get("collateral_status", "") or "").strip().lower() or None
            asset_stage = asset_row.get("verification_stage")
            asset_include = _safe_bool(asset_row.get("include_in_credit"), True)
            asset_note = asset_row.get("notes") or asset_row.get("comment")
            asset_update = asset_row.get("last_updated")

            if asset_value and loan_amount > 0:
                ltv = loan_amount / asset_value if asset_value else None
                if ltv is not None:
                    ltv_value = round(float(ltv), 4)
                    if ltv <= target_ltv:
                        factor *= min(1.25, 1.0 + (target_ltv - ltv) * 0.12)
                    else:
                        factor *= max(0.55, 1.0 - (ltv - target_ltv) * 0.25)

            status_factor_map = {
                "validated": 1.08,
                "approved": 1.08,
                "cleared": 1.05,
                "under_verification": 0.92,
                "under_validation": 0.92,
                "under_review": 0.9,
                "monitor": 0.95,
                "re_evaluate": 0.88,
                "reevaluate": 0.88,
                "re_evaluation": 0.88,
                "reinspection": 0.9,
                "denied_fraud": 0.65,
                "fraudulent": 0.65,
                "rejected": 0.75,
            }
            if asset_status in status_factor_map:
                factor *= status_factor_map[asset_status]

            if asset_confidence:
                factor *= 1.0 + max(-0.08, min(0.08, (asset_confidence - 0.8) * 0.15))
            if asset_legitimacy:
                factor *= 1.0 + max(-0.1, min(0.1, (asset_legitimacy - 0.85) * 0.2))

            if not asset_include or asset_status in {"denied_fraud", "fraudulent", "rejected"}:
                asset_override = "denied_asset_fraud"
                asset_include = False
                factor = min(factor, 0.6)
            elif asset_status in {"under_verification", "under_validation", "under_review", "re_evaluate", "re_evaluation", "reevaluate"}:
                asset_override = "pending_asset_review"

        ltv_values.append(ltv_value)
        factor = float(np.clip(factor, 0.5, 1.3))
        adjustment_factors.append(round(factor, 4))
        df.at[idx, "score"] = float(np.clip(base_score * factor, 0.0, 1.0))

        asset_values.append(asset_value if asset_value is not None else None)
        asset_confidences.append(asset_confidence if asset_confidence is not None else None)
        asset_legitimacies.append(asset_legitimacy if asset_legitimacy is not None else None)
        asset_statuses.append(asset_status)
        asset_stages.append(asset_stage)
        asset_includes.append(asset_include)
        asset_overrides.append(asset_override)
        asset_notes.append(asset_note)
        asset_updated.append(asset_update)

    df["asset_ltv"] = ltv_values
    df["asset_adjustment_factor"] = adjustment_factors
    df["asset_collateral_value"] = asset_values
    df["asset_confidence"] = asset_confidences
    df["asset_legitimacy_score"] = asset_legitimacies
    df["asset_collateral_status"] = asset_statuses
    df["asset_verification_stage"] = asset_stages
    df["asset_include_in_credit"] = asset_includes
    df["asset_decision_override"] = asset_overrides
    df["asset_notes"] = asset_notes
    df["asset_last_updated"] = asset_updated

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

        override = row.get("asset_decision_override")
        if override:
            row_reasons["asset_override"] = override
            if override == "denied_asset_fraud":
                final_decision = "denied_asset_fraud"
            elif override == "pending_asset_review" and final_decision == "approved":
                final_decision = "pending_asset_review"

        include_flag = row.get("asset_include_in_credit")
        if include_flag is False and final_decision == "approved":
            final_decision = "pending_asset_review"

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

    if asset_summary:
        summary["asset_bridge"] = asset_summary

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
        "asset_collateral_value",
        "asset_collateral_status",
        "asset_verification_stage",
        "asset_confidence",
        "asset_legitimacy_score",
        "asset_ltv",
        "asset_adjustment_factor",
        "asset_decision_override",
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
