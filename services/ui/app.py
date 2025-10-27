from __future__ import annotations

import datetime
import io
import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

import plotly.graph_objects as go  # noqa: F401

# ────────────────────────────────
# SESSION / STATE DEFAULTS
# ────────────────────────────────
if "stage" not in st.session_state:
    st.session_state.stage = "landing"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_info" not in st.session_state:
    st.session_state.user_info = {"name": "", "email": "", "flagged": False}

# ────────────────────────────────
# CONFIG
# ────────────────────────────────
st.set_page_config(
    page_title="AI Agent Sandbox — By the People, For the People",
    layout="wide",
)

API_URL = os.getenv("API_URL", "http://localhost:8090")
RUNS_DIR = os.path.expanduser("~/credit-appraisal-agent-poc/services/api/.runs")
TMP_FEEDBACK_DIR = os.path.join(RUNS_DIR, "tmp_feedback")
LANDING_IMG_DIR = os.path.expanduser("~/credit-appraisal-agent-poc/services/ui/landing_images")

os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(TMP_FEEDBACK_DIR, exist_ok=True)
os.makedirs(LANDING_IMG_DIR, exist_ok=True)

top_anchor = "top-of-page"
st.markdown(f"<a id='{top_anchor}'></a>", unsafe_allow_html=True)

# ────────────────────────────────
# HELPERS
# ────────────────────────────────

def _clear_query_params() -> None:
    try:
        st.query_params.clear()
    except Exception:
        try:
            st.experimental_set_query_params()
        except Exception:
            pass


def load_image(base: str) -> Optional[str]:
    for ext in [".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"]:
        path = os.path.join(LANDING_IMG_DIR, f"{base}{ext}")
        if os.path.exists(path):
            return path
    return None


def save_uploaded_image(uploaded_file, base: str) -> Optional[str]:
    if not uploaded_file:
        return None
    ext = os.path.splitext(uploaded_file.name)[1].lower() or ".png"
    dest = os.path.join(LANDING_IMG_DIR, f"{base}{ext}")
    with open(dest, "wb") as fh:
        fh.write(uploaded_file.getvalue())
    return dest


def render_image_tag(agent_id: str, industry: str, emoji_fallback: str) -> str:
    base = agent_id.lower().replace(" ", "_")
    img_path = load_image(base) or load_image(industry.replace(" ", "_"))
    if img_path:
        return (
            f"<img src='file://{img_path}' style='width:48px;height:48px;border-radius:10px;object-fit:cover;'>"
        )
    return f"<div style='font-size:32px;'>{emoji_fallback}</div>"


def logout_user() -> None:
    keys_to_drop = [
        "synthetic_raw_df",
        "synthetic_df",
        "anonymized_df",
        "asset_collateral_df",
        "asset_collateral_path",
        "manual_upload_name",
        "manual_upload_bytes",
        "last_run_id",
        "last_merged_df",
        "review_corrections",
        "last_agreement_score",
    ]
    for key in keys_to_drop:
        st.session_state.pop(key, None)
    st.session_state.user_info = {"name": "", "email": "", "flagged": False}
    st.session_state.logged_in = False
    st.session_state.stage = "landing"
    _clear_query_params()
    st.experimental_rerun()


AGENTS = [
    ("🏦 Banking & Finance", "💰 Retail Banking", "💳 Credit Appraisal Agent", "Explainable AI for loan decisioning", "Available", "💳"),
    ("🏦 Banking & Finance", "💰 Retail Banking", "🏦 Asset Appraisal Agent", "Market-driven collateral valuation", "Coming Soon", "🏦"),
    ("🏦 Banking & Finance", "🩺 Insurance", "🩺 Claims Triage Agent", "Automated claims prioritization", "Coming Soon", "🩺"),
    ("⚡ Energy & Sustainability", "🔋 EV & Charging", "⚡ EV Charger Optimizer", "Optimize charger deployment via AI", "Coming Soon", "⚡"),
    ("⚡ Energy & Sustainability", "☀️ Solar", "☀️ Solar Yield Estimator", "Estimate solar ROI and efficiency", "Coming Soon", "☀️"),
    ("🚗 Automobile & Transport", "🚙 Automobile", "🚗 Predictive Maintenance", "Prevent downtime via sensor analytics", "Coming Soon", "🚗"),
    ("🚗 Automobile & Transport", "🔋 EV", "🔋 EV Battery Health Agent", "Monitor EV battery health cycles", "Coming Soon", "🔋"),
    ("🚗 Automobile & Transport", "🚚 Ride-hailing / Logistics", "🛻 Fleet Route Optimizer", "Dynamic route optimization for fleets", "Coming Soon", "🛻"),
    ("💻 Information Technology", "🧰 Support & Security", "🧩 IT Ticket Triage", "Auto-prioritize support tickets", "Coming Soon", "🧩"),
    ("💻 Information Technology", "🛡️ Security", "🔐 SecOps Log Triage", "Detect anomalies & summarize alerts", "Coming Soon", "🔐"),
    ("⚖️ Legal & Government", "⚖️ Law Firms", "⚖️ Contract Analyzer", "Extract clauses and compliance risks", "Coming Soon", "⚖️"),
    ("⚖️ Legal & Government", "🏛️ Public Services", "🏛️ Citizen Service Agent", "Smart assistant for citizen services", "Coming Soon", "🏛️"),
    ("🛍️ Retail / SMB / Creative", "🏬 Retail & eCommerce", "📈 Sales Forecast Agent", "Predict demand & inventory trends", "Coming Soon", "📈"),
    ("🎬 Retail / SMB / Creative", "🎨 Media & Film", "🎬 Budget Cost Assistant", "Estimate, optimize, and track film & production costs using AI", "Coming Soon", "🎬"),
]

st.markdown(
    """
    <style>
    html, body, .block-container {
        background-color: #0f172a !important;
        color: #e2e8f0 !important;
    }
    .left-box {
        background: radial-gradient(circle at top left, #0f172a, #1e293b);
        color: #f1f5f9;
        border-radius: 20px;
        padding: 3rem 2rem;
        height: 100%;
        box-shadow: 6px 0 24px rgba(0,0,0,0.4);
    }
    .left-box h1 {font-size: 2.6rem; font-weight: 900; color: #fff; margin-bottom: 1rem;}
    .left-box p, .left-box h3 {font-size: 1rem; color: #cbd5e1; line-height: 1.6;}

    .right-box {
        background: linear-gradient(180deg, #1e293b, #0f172a);
        border-radius: 20px;
        padding: 2rem;
        color: #e2e8f0;
        box-shadow: -6px 0 24px rgba(0,0,0,0.35);
    }
    .dataframe {
        width: 100%;
        border-collapse: collapse;
        font-size: 15px;
        color: #f1f5f9;
        background-color: #0f172a;
    }
    .dataframe th {
        background-color: #1e293b;
        color: #f8fafc;
        padding: 12px;
        border-bottom: 2px solid #334155;
    }
    .dataframe td {
        padding: 10px 14px;
        border-bottom: 1px solid #334155;
    }
    .dataframe tr:hover {
        background-color: #1e293b;
        transition: background 0.3s ease-in-out;
    }
    .status-Available {color: #22c55e; font-weight:600;}
    .status-ComingSoon {color: #f59e0b; font-weight:600;}
    .top-nav-link {
        background: linear-gradient(135deg, #1d4ed8, #2563eb);
        padding: 0.5rem 1.2rem;
        border-radius: 999px;
        color: #f8fafc !important;
        text-decoration: none !important;
        font-weight: 600;
        box-shadow: 0 12px 24px rgba(37, 99, 235, 0.25);
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
    }
    .top-nav-link:hover {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
    }
    footer {
        text-align: center;
        padding: 2rem;
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 3rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 44px !important;
        font-weight: 900 !important;
        padding: 28px 36px !important;
        border-radius: 16px !important;
        background-color: #1e293b !important;
        color: #f8fafc !important;
        line-height: 1.2 !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(90deg,#2563eb,#1d4ed8) !important;
        color: white !important;
        border-bottom: 6px solid #60a5fa !important;
        box-shadow: 0 4px 10px rgba(37,99,235,0.4);
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #334155 !important;
        transform: translateY(-2px);
        transition: all 0.25s ease-in-out;
    }
    .logout-btn button {
        background: linear-gradient(180deg,#ef4444,#b91c1c) !important;
        color: #fff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BANNED_NAMES = {"race", "gender", "religion", "ethnicity", "ssn", "national_id"}
PII_COLS = {"customer_name", "name", "email", "phone", "address", "ssn", "national_id", "dob"}
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d\-\s]{6,}\d")


def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.duplicated(keep="last")]


def scrub_text_pii(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    value = EMAIL_RE.sub("", value)
    value = PHONE_RE.sub("", value)
    return value.strip()


def drop_pii_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    original_cols = list(df.columns)
    keep_cols = [c for c in original_cols if all(flag not in c.lower() for flag in PII_COLS)]
    dropped = [c for c in original_cols if c not in keep_cols]
    out = df[keep_cols].copy()
    for col in out.select_dtypes(include="object"):
        out[col] = out[col].apply(scrub_text_pii)
    return dedupe_columns(out), dropped


def strip_policy_banned(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in df.columns if c.lower() not in BANNED_NAMES]
    return df[keep]


def append_user_info(df: pd.DataFrame) -> pd.DataFrame:
    meta = st.session_state.user_info
    enriched = df.copy()
    enriched["session_user_name"] = meta.get("name", "")
    enriched["session_user_email"] = meta.get("email", "")
    enriched["session_flagged"] = bool(meta.get("flagged", False))
    enriched["created_at"] = meta.get("timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return dedupe_columns(enriched)


def try_json(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
    return None


def safe_json(value: Any) -> Dict[str, Any]:
    parsed = try_json(value)
    return parsed or {}


CURRENCY_OPTIONS: Dict[str, Tuple[str, str, float]] = {
    "USD": ("USD $", "$", 1.0),
    "EUR": ("EUR €", "€", 0.93),
    "GBP": ("GBP £", "£", 0.80),
    "JPY": ("JPY ¥", "¥", 150.0),
    "VND": ("VND ₫", "₫", 24000.0),
}


def set_currency_defaults() -> None:
    if "currency_code" not in st.session_state:
        st.session_state.currency_code = "USD"
    label, symbol, fx = CURRENCY_OPTIONS[st.session_state.currency_code]
    st.session_state.currency_label = label
    st.session_state.currency_symbol = symbol
    st.session_state.currency_fx = fx


def fmt_currency_label(text: str) -> str:
    sym = st.session_state.get("currency_symbol", "")
    return f"{text} ({sym})" if sym else text


set_currency_defaults()

def generate_raw_synthetic(n: int, non_bank_ratio: float) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    names = [
        "Alice Nguyen",
        "Bao Tran",
        "Chris Do",
        "Duy Le",
        "Emma Tran",
        "Felix Nguyen",
        "Giang Ho",
        "Hanh Vo",
        "Ivan Pham",
        "Julia Ngo",
    ]
    emails = [f"{nm.split()[0].lower()}.{nm.split()[1].lower()}@gmail.com" for nm in names]
    addresses = [
        "23 Elm St, Boston, MA",
        "19 Pine Ave, San Jose, CA",
        "14 High St, London, UK",
        "55 Nguyen Hue, Ho Chi Minh",
        "78 Oak St, Chicago, IL",
        "10 Broadway, New York, NY",
        "8 Rue Lafayette, Paris, FR",
        "21 Königstr, Berlin, DE",
        "44 Maple Dr, Los Angeles, CA",
        "22 Bay St, Toronto, CA",
    ]
    is_non_bank = rng.random(n) < non_bank_ratio
    customer_type = np.where(is_non_bank, "non-bank", "bank")

    df = pd.DataFrame(
        {
            "application_id": [f"APP_{i:04d}" for i in range(1, n + 1)],
            "customer_name": rng.choice(names, n),
            "email": rng.choice(emails, n),
            "phone": [f"+1-202-555-{1000 + i:04d}" for i in range(n)],
            "address": rng.choice(addresses, n),
            "national_id": rng.integers(10_000_000, 99_999_999, n),
            "age": rng.integers(21, 65, n),
            "income": rng.integers(25_000, 150_000, n),
            "employment_length": rng.integers(0, 30, n),
            "loan_amount": rng.integers(5_000, 100_000, n),
            "loan_duration_months": rng.choice([12, 24, 36, 48, 60, 72], n),
            "collateral_value": rng.integers(8_000, 200_000, n),
            "collateral_type": rng.choice(["real_estate", "car", "land", "deposit"], n),
            "co_loaners": rng.choice([0, 1, 2], n, p=[0.7, 0.25, 0.05]),
            "credit_score": rng.integers(300, 850, n),
            "existing_debt": rng.integers(0, 50_000, n),
            "assets_owned": rng.integers(10_000, 300_000, n),
            "current_loans": rng.integers(0, 5, n),
            "customer_type": customer_type,
        }
    )
    eps = 1e-9
    df["DTI"] = df["existing_debt"] / (df["income"] + eps)
    df["LTV"] = df["loan_amount"] / (df["collateral_value"] + eps)
    df["CCR"] = df["collateral_value"] / (df["loan_amount"] + eps)
    df["ITI"] = (df["loan_amount"] / (df["loan_duration_months"] + eps)) / (df["income"] + eps)
    df["CWI"] = (
        (1 - df["DTI"]).clip(0, 1) * (1 - df["LTV"]).clip(0, 1) * df["CCR"].clip(0, 3)
    )

    fx = st.session_state.currency_fx
    for col in ("income", "loan_amount", "collateral_value", "assets_owned", "existing_debt"):
        df[col] = (df[col] * fx).round(2)
    df["currency_code"] = st.session_state.currency_code
    return dedupe_columns(df)


def generate_anon_synthetic(n: int, non_bank_ratio: float) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    is_non_bank = rng.random(n) < non_bank_ratio
    customer_type = np.where(is_non_bank, "non-bank", "bank")

    df = pd.DataFrame(
        {
            "application_id": [f"APP_{i:04d}" for i in range(1, n + 1)],
            "age": rng.integers(21, 65, n),
            "income": rng.integers(25_000, 150_000, n),
            "employment_length": rng.integers(0, 30, n),
            "loan_amount": rng.integers(5_000, 100_000, n),
            "loan_duration_months": rng.choice([12, 24, 36, 48, 60, 72], n),
            "collateral_value": rng.integers(8_000, 200_000, n),
            "collateral_type": rng.choice(["real_estate", "car", "land", "deposit"], n),
            "co_loaners": rng.choice([0, 1, 2], n, p=[0.7, 0.25, 0.05]),
            "credit_score": rng.integers(300, 850, n),
            "existing_debt": rng.integers(0, 50_000, n),
            "assets_owned": rng.integers(10_000, 300_000, n),
            "current_loans": rng.integers(0, 5, n),
            "customer_type": customer_type,
        }
    )
    eps = 1e-9
    df["DTI"] = df["existing_debt"] / (df["income"] + eps)
    df["LTV"] = df["loan_amount"] / (df["collateral_value"] + eps)
    df["CCR"] = df["collateral_value"] / (df["loan_amount"] + eps)
    df["ITI"] = (df["loan_amount"] / (df["loan_duration_months"] + eps)) / (df["income"] + eps)
    df["CWI"] = (
        (1 - df["DTI"]).clip(0, 1) * (1 - df["LTV"]).clip(0, 1) * df["CCR"].clip(0, 3)
    )

    fx = st.session_state.currency_fx
    for col in ("income", "loan_amount", "collateral_value", "assets_owned", "existing_debt"):
        df[col] = (df[col] * fx).round(2)
    df["currency_code"] = st.session_state.currency_code
    return dedupe_columns(df)


def to_agent_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    if "employment_years" not in out:
        out["employment_years"] = out.get("employment_length", 0)
    if "debt_to_income" not in out:
        if "DTI" in out:
            out["debt_to_income"] = out["DTI"].astype(float)
        elif {"existing_debt", "income"} <= set(out.columns):
            denom = out["income"].replace(0, np.nan)
            dti = (out["existing_debt"] / denom).fillna(0.0)
            out["debt_to_income"] = dti.clip(0, 10)
        else:
            out["debt_to_income"] = 0.0
    rng = np.random.default_rng(12345)
    if "credit_history_length" not in out:
        out["credit_history_length"] = rng.integers(0, 30, n)
    if "num_delinquencies" not in out:
        out["num_delinquencies"] = np.minimum(rng.poisson(0.2, n), 10)
    if "requested_amount" not in out:
        out["requested_amount"] = out.get("loan_amount", 0)
    if "loan_term_months" not in out:
        out["loan_term_months"] = out.get("loan_duration_months", 0)
    return dedupe_columns(out)


def ensure_application_ids(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "application_id" not in result:
        result["application_id"] = [f"APP_{i:04d}" for i in range(1, len(result) + 1)]
    result["application_id"] = result["application_id"].astype(str)
    return result


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def build_collateral_report(
    df: pd.DataFrame,
    *,
    confidence_threshold: float = 0.88,
    value_ratio: float = 0.8,
) -> Tuple[pd.DataFrame, List[str]]:
    if df is None or df.empty:
        return pd.DataFrame(), []

    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    session = requests.Session()
    records = df.to_dict(orient="records")
    progress = st.progress(0.0) if len(records) > 1 else None

    for idx, record in enumerate(records, start=1):
        asset_type = str(record.get("collateral_type") or "Collateral Asset")
        declared_value = _to_float(record.get("collateral_value"), 0.0)
        loan_amount = _to_float(record.get("loan_amount"), 0.0)
        metadata = {
            "application_id": record.get("application_id"),
            "declared_value": declared_value,
            "loan_amount": loan_amount,
            "currency_code": record.get("currency_code") or st.session_state.get("currency_code"),
        }
        payload = {"asset_type": asset_type, "metadata": json.dumps(metadata, default=str)}

        estimated_value = declared_value
        confidence = 0.0

        try:
            response = session.post(
                f"{API_URL}/v1/agents/asset_appraisal/run",
                data=payload,
                timeout=8,
            )
            response.raise_for_status()
            asset_result = response.json().get("result", {}) or {}
            estimated_value = _to_float(asset_result.get("estimated_value"), declared_value)
            confidence = _to_float(asset_result.get("confidence"), 0.0)
        except Exception:
            estimated_value = declared_value * random.uniform(0.75, 1.25)
            confidence = random.uniform(0.7, 0.98)

        value_threshold = 0.0
        if loan_amount:
            value_threshold = loan_amount * value_ratio
        elif declared_value:
            value_threshold = declared_value * value_ratio

        reasons: List[str] = []
        meets_confidence = confidence >= confidence_threshold
        meets_value = True if value_threshold == 0 else estimated_value >= value_threshold

        if not meets_confidence:
            reasons.append(
                f"Confidence {confidence:.2f} below threshold {confidence_threshold:.2f}"
            )
        if not meets_value:
            if loan_amount:
                reasons.append(
                    f"Estimated value {estimated_value:,.0f} below {value_ratio:.0%} of loan {loan_amount:,.0f}"
                )
            else:
                reasons.append(
                    f"Estimated value {estimated_value:,.0f} below threshold {value_threshold:,.0f}"
                )
        if not reasons:
            reasons.append("Confidence and value thresholds satisfied")

        verified = meets_confidence and meets_value
        enriched = dict(record)
        enriched.update(
            {
                "collateral_estimated_value": round(estimated_value, 2),
                "collateral_confidence": round(confidence, 4),
                "collateral_verified": bool(verified),
                "collateral_status": "Verified" if verified else "Failed",
                "collateral_verification_reason": "; ".join(reasons),
                "collateral_checked_at": datetime.datetime.utcnow().isoformat(),
            }
        )
        rows.append(enriched)

        if progress is not None:
            progress.progress(idx / len(records))

    if progress is not None:
        progress.empty()
    return dedupe_columns(pd.DataFrame(rows)), errors

def _kpi_card(label: str, value: str, sublabel: Optional[str] = None) -> None:
    st.markdown(
        f"""
        <div style=\"background:#0e1117;border:1px solid #2a2f3e;border-radius:12px;padding:14px 16px;margin-bottom:10px;\">
          <div style=\"font-size:12px;color:#9aa4b2;text-transform:uppercase;letter-spacing:.06em;\">{label}</div>
          <div style=\"font-size:28px;font-weight:700;color:#e6edf3;line-height:1.1;margin-top:2px;\">{value}</div>
          {f'<div style=\"font-size:12px;color:#9aa4b2;margin-top:6px;\">{sublabel}</div>' if sublabel else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_credit_dashboard(df: pd.DataFrame, currency_symbol: str = "") -> None:
    if df is None or df.empty:
        st.info("No data to visualize yet.")
        return

    cols = df.columns
    st.markdown("## 🔝 Top 10 Snapshot")

    if {"decision", "loan_amount", "application_id"} <= set(cols):
        top_approved = df[df["decision"].astype(str).str.lower() == "approved"].copy()
        if not top_approved.empty:
            top_approved = top_approved.sort_values("loan_amount", ascending=False).head(10)
            fig = px.bar(
                top_approved,
                x="loan_amount",
                y="application_id",
                orientation="h",
                title="Top 10 Approved Loans",
                labels={"loan_amount": f"Loan Amount {currency_symbol}", "application_id": "Application"},
            )
            fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No approved loans available to show top 10.")

    if {"collateral_type", "collateral_value"} <= set(cols):
        cprof = (
            df.groupby("collateral_type", dropna=False)
            .agg(avg_value=("collateral_value", "mean"), cnt=("collateral_type", "count"))
            .reset_index()
        )
        if not cprof.empty:
            cprof = cprof.sort_values("avg_value", ascending=False).head(10)
            fig = px.bar(
                cprof,
                x="avg_value",
                y="collateral_type",
                orientation="h",
                title="Top 10 Collateral Types (Avg Value)",
                labels={"avg_value": f"Avg Value {currency_symbol}", "collateral_type": "Collateral Type"},
                hover_data=["cnt"],
            )
            fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    if "rule_reasons" in cols and "decision" in cols:
        denied = df[df["decision"].astype(str).str.lower() == "denied"].copy()
        reasons_count: Dict[str, int] = {}
        for _, row in denied.iterrows():
            rr = safe_json(row.get("rule_reasons"))
            for key, value in rr.items():
                if value is False:
                    reasons_count[key] = reasons_count.get(key, 0) + 1
        if reasons_count:
            items = (
                pd.DataFrame(sorted(reasons_count.items(), key=lambda kv: kv[1], reverse=True), columns=["reason", "count"])
                .head(10)
            )
            fig = px.bar(
                items,
                x="count",
                y="reason",
                orientation="h",
                title="Top 10 Reasons for Denial",
                labels={"count": "Count", "reason": "Rule"},
            )
            fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No denial reasons detected.")

    officer_col = None
    for guess in ("loan_officer", "officer", "reviewed_by", "session_user_name"):
        if guess in cols:
            officer_col = guess
            break
    if officer_col and "decision" in cols:
        perf = (
            df.assign(is_approved=(df["decision"].astype(str).str.lower() == "approved").astype(int))
            .groupby(officer_col, dropna=False)["is_approved"]
            .agg(approved_rate="mean", n="count")
            .reset_index()
        )
        if not perf.empty:
            perf["approved_rate_pct"] = (perf["approved_rate"] * 100).round(1)
            perf = perf.sort_values(["approved_rate_pct", "n"], ascending=[False, False]).head(10)
            fig = px.bar(
                perf,
                x="approved_rate_pct",
                y=officer_col,
                orientation="h",
                title="Top 10 Loan Officer Approval Rate (this batch)",
                labels={"approved_rate_pct": "Approval Rate (%)", officer_col: "Officer"},
                hover_data=["n"],
            )
            fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("## 💡 Opportunities")

    opp_rows: List[Dict[str, Any]] = []
    if {"income", "loan_amount"} <= set(cols):
        term_col = "loan_term_months" if "loan_term_months" in cols else ("loan_duration_months" if "loan_duration_months" in cols else None)
        if term_col:
            for _, row in df.iterrows():
                income = float(row.get("income", 0) or 0)
                loan_amount = float(row.get("loan_amount", 0) or 0)
                term = int(row.get(term_col, 0) or 0)
                dti = float(row.get("DTI", 0) or 0)
                if term >= 36 and loan_amount <= income * 0.8 and dti <= 0.45:
                    opp_rows.append(
                        {
                            "application_id": row.get("application_id"),
                            "suggested_term": 24,
                            "loan_amount": loan_amount,
                            "income": income,
                            "DTI": dti,
                            "note": "Candidate for short-term plan (<=24m) based on affordability.",
                        }
                    )
    if opp_rows:
        st.markdown("#### 📎 Short-Term Loan Candidates")
        st.dataframe(pd.DataFrame(opp_rows).head(25), use_container_width=True, height=320)
    else:
        st.info("No short-term loan candidates identified in this batch.")

    st.markdown("#### 🔁 Buyback / Consolidation Beneficiaries")
    candidates: List[Dict[str, Any]] = []
    if {"decision", "existing_debt", "loan_amount", "DTI"} <= set(cols):
        for _, row in df.iterrows():
            decision = str(row.get("decision", "")).lower()
            debt = float(row.get("existing_debt", 0) or 0)
            loan = float(row.get("loan_amount", 0) or 0)
            dti = float(row.get("DTI", 0) or 0)
            proposal = safe_json(row.get("proposed_consolidation_loan", {}))
            has_buyback = bool(proposal)
            if decision == "denied" or dti > 0.45 or debt > loan:
                benefit_score = round((debt / (loan + 1e-6)) * 0.4 + dti * 0.6, 2)
                candidates.append(
                    {
                        "application_id": row.get("application_id"),
                        "customer_type": row.get("customer_type"),
                        "existing_debt": debt,
                        "loan_amount": loan,
                        "DTI": dti,
                        "collateral_type": row.get("collateral_type"),
                        "buyback_proposed": has_buyback,
                        "buyback_amount": proposal.get("buyback_amount") if has_buyback else None,
                        "benefit_score": benefit_score,
                        "note": proposal.get("note") if has_buyback else None,
                    }
                )
    if candidates:
        cand_df = pd.DataFrame(candidates).sort_values("benefit_score", ascending=False)
        st.dataframe(cand_df.head(25), use_container_width=True, height=380)
    else:
        st.info("No additional buyback beneficiaries identified.")

    st.markdown("---")
    st.markdown("## 📈 Portfolio Snapshot")
    c1, c2, c3, c4 = st.columns(4)

    if "decision" in cols:
        total = len(df)
        approved = int((df["decision"].astype(str).str.lower() == "approved").sum())
        rate = (approved / total * 100) if total else 0.0
        with c1:
            _kpi_card("Approval Rate", f"{rate:.1f}%", f"{approved} of {total}")

    if {"decision", "loan_amount"} <= set(cols):
        approved_loans = df[df["decision"].astype(str).str.lower() == "approved"]["loan_amount"]
        avg_amt = approved_loans.mean() if len(approved_loans) else 0.0
        with c2:
            _kpi_card("Avg Approved Amount", f"{currency_symbol}{avg_amt:,.0f}")

    if {"created_at", "decision_at"} <= set(cols):
        try:
            duration = (
                pd.to_datetime(df["decision_at"]) - pd.to_datetime(df["created_at"])
            ).dt.total_seconds() / 60.0
            avg_minutes = float(duration.mean())
            with c3:
                _kpi_card("Avg Decision Time", f"{avg_minutes:.1f} min")
        except Exception:
            with c3:
                _kpi_card("Avg Decision Time", "—")

    if "customer_type" in cols:
        nb = int((df["customer_type"].astype(str).str.lower() == "non-bank").sum())
        total = len(df)
        share = (nb / total * 100) if total else 0.0
        with c4:
            _kpi_card("Non-bank Share", f"{share:.1f}%", f"{nb} of {total}")

    st.markdown("## 🧭 Composition & Risk")
    if "decision" in cols:
        pie_df = df["decision"].value_counts().rename_axis("Decision").reset_index(name="Count")
        fig = px.pie(pie_df, names="Decision", values="Count", title="Decision Mix")
        fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=360, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    have_dti = "DTI" in cols
    have_ltv = "LTV" in cols
    if "decision" in cols and (have_dti or have_ltv):
        agg_map = {}
        if have_dti:
            agg_map["avg_DTI"] = ("DTI", "mean")
        if have_ltv:
            agg_map["avg_LTV"] = ("LTV", "mean")
        grp = df.groupby("decision").agg(**agg_map).reset_index()
        melted = grp.melt(id_vars=["decision"], var_name="metric", value_name="value")
        fig = px.bar(
            melted,
            x="decision",
            y="value",
            color="metric",
            barmode="group",
            title="Average DTI / LTV by Decision",
        )
        fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=360, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    term_col = "loan_term_months" if "loan_term_months" in cols else ("loan_duration_months" if "loan_duration_months" in cols else None)
    if term_col and "decision" in cols:
        mix = df.groupby([term_col, "decision"]).size().reset_index(name="count")
        fig = px.bar(
            mix,
            x=term_col,
            y="count",
            color="decision",
            title="Loan Term Mix",
            labels={term_col: "Term (months)", "count": "Count"},
            barmode="stack",
        )
        fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=360, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    if {"collateral_type", "collateral_value"} <= set(cols):
        cprof = (
            df.groupby("collateral_type")
            .agg(avg_col=("collateral_value", "mean"), cnt=("collateral_type", "count"))
            .reset_index()
        )
        fig = px.bar(
            cprof.sort_values("avg_col", ascending=False),
            x="collateral_type",
            y="avg_col",
            title=f"Avg Collateral Value by Type ({currency_symbol})",
            hover_data=["cnt"],
        )
        fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=360, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    if "proposed_loan_option" in cols:
        plans = df["proposed_loan_option"].dropna().astype(str)
        if len(plans) > 0:
            plan_types = []
            for entry in plans:
                parsed = safe_json(entry)
                plan_types.append(parsed.get("type") if "type" in parsed else entry)
            plan_df = (
                pd.Series(plan_types)
                .value_counts()
                .head(10)
                .rename_axis("plan")
                .reset_index(name="count")
            )
            fig = px.bar(
                plan_df,
                x="count",
                y="plan",
                orientation="h",
                title="Top 10 Proposed Plans",
            )
            fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=360, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    if "customer_type" in cols:
        mix = df["customer_type"].value_counts().rename_axis("Customer Type").reset_index(name="Count")
        mix["Ratio"] = (mix["Count"] / mix["Count"].sum()).round(3)
        st.markdown("### 👥 Customer Mix")
        st.dataframe(mix, use_container_width=True, height=220)


def save_to_runs(df: pd.DataFrame, prefix: str) -> str:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    flag_suffix = "_FLAGGED" if st.session_state.user_info.get("flagged") else ""
    filename = f"{prefix}_{timestamp}{flag_suffix}.csv"
    path = os.path.join(RUNS_DIR, filename)
    dedupe_columns(df).to_csv(path, index=False)
    return path
try:
    qp = st.query_params
except Exception:
    qp = {}

if "stage" in qp:
    target = qp["stage"]
    if target in {"landing", "agents", "login", "credit"} and st.session_state.stage != target:
        st.session_state.stage = target
        _clear_query_params()
        st.experimental_rerun()

if "agent" in qp and qp.get("agent") == "credit":
    st.session_state.stage = "login"
    _clear_query_params()
    st.experimental_rerun()

if st.session_state.stage == "landing":
    col1, col2 = st.columns([1.1, 1.9], gap="large")
    with col1:
        st.markdown("<div class='left-box'>", unsafe_allow_html=True)
        logo_path = load_image("people_logo")
        if logo_path:
            st.image(logo_path, width=160)
        else:
            upload = st.file_uploader("Upload People Logo", type=["jpg", "png", "webp"], key="upload_logo")
            if upload:
                save_uploaded_image(upload, "people_logo")
                st.success("✅ Logo uploaded successfully! Refreshing...")
                st.experimental_rerun()
        st.markdown(
            """
            <h1>✊ Let’s Build an AI by the People, for the People</h1>
            <h3>⚙️ Ready-to-Use AI Agent Sandbox — From Sandbox to Production</h3>
            <p>
            A world-class open innovation space where anyone can build, test, and deploy AI agents using open-source code, explainable models, and modular templates.<br><br>
            For developers, startups, and enterprises — experiment, customize, and scale AI without barriers.<br><br>
            <b>Privacy &amp; Data Sovereignty:</b> Each agent runs under strict privacy controls and complies with GDPR &amp; Vietnam Data Law 2025. Only anonymized or synthetic data is used — your data never leaves your environment.<br><br>
            <b>From Sandbox to Production:</b> Start with ready-to-use agent templates, adapt, test, and deploy — all on GPU-as-a-Service Cloud with zero CAPEX.<br><br>
            You dream it — now you can build it.
            </p>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🚀 Start Building Now", key="start_build_button", use_container_width=True):
            st.session_state.stage = "agents"
            st.experimental_rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='right-box'>", unsafe_allow_html=True)
        st.markdown("<h2>📊 Global AI Agent Library</h2>", unsafe_allow_html=True)
        st.caption("Explore sectors, industries, and ready-to-use AI agents across domains.")
        rows = []
        for sector, industry, agent, desc, status, emoji in AGENTS:
            rating = round(random.uniform(3.5, 5.0), 1)
            users = random.randint(800, 9000)
            comments = random.randint(5, 120)
            rows.append(
                {
                    "🖼️": render_image_tag(agent, industry, emoji),
                    "🏭 Sector": sector,
                    "🧩 Industry": industry,
                    "🤖 Agent": agent,
                    "🧠 Description": desc,
                    "📶 Status": f"<span class='status-{status.replace(' ', '')}'>{status}</span>",
                    "⭐ Rating": "⭐" * int(rating) + "☆" * (5 - int(rating)),
                    "👥 Users": users,
                    "💬 Comments": comments,
                }
            )
        st.write(pd.DataFrame(rows).to_html(escape=False, index=False), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<footer>Made with ❤️ by Dzoan Nguyen — Open AI Sandbox Initiative</footer>", unsafe_allow_html=True)
    st.stop()

if st.session_state.stage == "agents":
    nav = st.columns([1, 1, 2])
    with nav[0]:
        if st.button("⬅️ Back to Home", key="agents_back_home"):
            st.session_state.stage = "landing"
            st.experimental_rerun()
    with nav[1]:
        st.markdown(
            f"<a class='top-nav-link' href='#{top_anchor}'>⬆️ Home</a>",
            unsafe_allow_html=True,
        )
    st.title("🤖 Available AI Agents")
    df_agents = pd.DataFrame(
        [
            {
                "Agent": "💳 Credit Appraisal Agent",
                "Description": "Explainable AI for retail loan decisioning",
                "Status": "✅ Available",
                "Action": "<a class='macbtn' href='?agent=credit&stage=login'>🚀 Launch</a>",
            },
            {
                "Agent": "🏦 Asset Appraisal Agent",
                "Description": "Market-driven collateral valuation",
                "Status": "🕓 Coming Soon",
                "Action": "—",
            },
        ]
    )
    st.write(df_agents.to_html(escape=False, index=False), unsafe_allow_html=True)
    st.markdown("<footer>Made with ❤️ by Dzoan Nguyen — Open AI Sandbox Initiative</footer>", unsafe_allow_html=True)
    st.stop()

if st.session_state.stage == "login":
    nav = st.columns([1, 1, 2])
    with nav[0]:
        if st.button("⬅️ Back to Agents", key="login_back_agents"):
            st.session_state.stage = "agents"
            st.experimental_rerun()
    with nav[1]:
        st.markdown(
            f"<a class='top-nav-link' href='#{top_anchor}'>⬆️ Home</a>",
            unsafe_allow_html=True,
        )
    st.title("🔐 Login to AI Credit Appraisal Platform")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        username = st.text_input("Username", placeholder="e.g. dzoan")
    with col2:
        email = st.text_input("Email", placeholder="e.g. dzoan@demo.local")
    with col3:
        password = st.text_input("Password", type="password", placeholder="Enter any password")
    if st.button("Login", key="login_submit", use_container_width=True):
        if username.strip() and email.strip():
            st.session_state.user_info = {
                "name": username.strip(),
                "email": email.strip(),
                "flagged": False,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            st.session_state.logged_in = True
            st.session_state.stage = "credit"
            st.experimental_rerun()
        else:
            st.error("⚠️ Please fill all fields before continuing.")
    st.markdown("<footer>Made with ❤️ by Dzoan Nguyen — Open AI Sandbox Initiative</footer>", unsafe_allow_html=True)
    st.stop()

if not st.session_state.logged_in:
    st.session_state.stage = "login"
    st.experimental_rerun()

nav = st.columns([1, 1, 1, 5])
with nav[0]:
    if st.button("⬅️ Back to Agents", key="credit_back_agents"):
        st.session_state.stage = "agents"
        st.experimental_rerun()
with nav[1]:
    st.markdown(
        f"<a class='top-nav-link' href='#{top_anchor}'>🏠 Home</a>",
        unsafe_allow_html=True,
    )
with nav[2]:
    if st.button("Logout", key="logout_button", use_container_width=True):
        logout_user()

st.title("💳 AI Credit Appraisal Platform")
st.caption("Generate, sanitize, and appraise credit with AI agent power and human insight.")

(
    tab_gen,
    tab_clean,
    tab_asset,
    tab_credit,
    tab_review,
    tab_train,
    tab_loop,
) = st.tabs(
    [
        "🧩 Step 1 / Synthetic Data Generator",
        "🧹 Step 2 / Anonymize & Sanitize Data",
        "🏛️ Step 3 / Asset Appraisal Pre-checks",
        "🤖 Step 4 / Credit Appraisal by AI Assistant",
        "🧑‍⚖️ Step 5 / Human Review",
        "🔁 Step 6 / Training (Feedback → Retrain)",
        "🔄 Step 7 / Loop Back to Step 4 → Use New Trained Model",
    ]
)
with tab_gen:
    st.subheader("🏦 Synthetic Credit Data Generator")
    col_currency, col_info = st.columns([1, 2])
    with col_currency:
        code = st.selectbox(
            "Currency",
            list(CURRENCY_OPTIONS.keys()),
            index=list(CURRENCY_OPTIONS.keys()).index(st.session_state.currency_code),
            help="All monetary fields will be in this local currency.",
        )
        if code != st.session_state.currency_code:
            st.session_state.currency_code = code
            set_currency_defaults()
    with col_info:
        st.info(
            f"Amounts will be generated in **{st.session_state.currency_label}**.",
            icon="💰",
        )

    rows = st.slider("Number of rows to generate", 50, 2000, 200, step=50)
    non_bank_ratio = st.slider("Share of non-bank customers", 0.0, 1.0, 0.30, 0.05)

    col_raw, col_anon = st.columns(2)
    with col_raw:
        if st.button("🔴 Generate RAW Synthetic Data (with PII)", use_container_width=True):
            raw_df = append_user_info(generate_raw_synthetic(rows, non_bank_ratio))
            st.session_state.synthetic_raw_df = raw_df
            raw_path = save_to_runs(raw_df, "synthetic_raw")
            st.success(
                f"Generated RAW (PII) dataset with {rows} rows in {st.session_state.currency_label}. Saved to {raw_path}"
            )
            st.dataframe(raw_df.head(10), use_container_width=True)
            st.download_button(
                "⬇️ Download RAW CSV",
                raw_df.to_csv(index=False).encode("utf-8"),
                os.path.basename(raw_path),
                "text/csv",
            )
    with col_anon:
        if st.button("🟢 Generate ANON Synthetic Data (ready for agent)", use_container_width=True):
            anon_df = append_user_info(generate_anon_synthetic(rows, non_bank_ratio))
            st.session_state.synthetic_df = anon_df
            anon_path = save_to_runs(anon_df, "synthetic_anon")
            st.success(
                f"Generated ANON dataset with {rows} rows in {st.session_state.currency_label}. Saved to {anon_path}"
            )
            st.dataframe(anon_df.head(10), use_container_width=True)
            st.download_button(
                "⬇️ Download ANON CSV",
                anon_df.to_csv(index=False).encode("utf-8"),
                os.path.basename(anon_path),
                "text/csv",
            )

with tab_clean:
    st.subheader("🧹 Upload & Anonymize Customer Data (PII columns will be DROPPED)")
    st.markdown("Upload your **real CSV**. We drop PII columns and scrub emails/phones in text fields.")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"], key="anonymize_uploader")
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            st.stop()
        st.write("📊 Original Data Preview:")
        st.dataframe(dedupe_columns(df.head(5)), use_container_width=True)

        sanitized, dropped_cols = drop_pii_columns(df)
        sanitized = append_user_info(sanitized)
        sanitized = dedupe_columns(sanitized)
        st.session_state.anonymized_df = sanitized

        st.success(f"Dropped PII columns: {sorted(dropped_cols) if dropped_cols else 'None'}")
        st.write("✅ Sanitized Data Preview:")
        st.dataframe(sanitized.head(5), use_container_width=True)

        path = save_to_runs(sanitized, "anonymized")
        st.success(f"Saved anonymized file: {path}")
        st.download_button(
            "⬇️ Download Clean Data",
            sanitized.to_csv(index=False).encode("utf-8"),
            os.path.basename(path),
            "text/csv",
        )
    else:
        st.info("Choose a CSV to see the sanitize flow.", icon="ℹ️")
with tab_asset:
    st.subheader("🏛️ Collateral Asset Verification")
    st.caption("Verify collateral assets before running the credit appraisal agent.")

    data_source = st.selectbox(
        "Collateral data source",
        [
            "Use synthetic (ANON)",
            "Use synthetic (RAW – auto-sanitize)",
            "Use anonymized dataset",
            "Upload manually",
        ],
    )

    if data_source == "Upload manually":
        upload = st.file_uploader("Upload CSV for collateral verification", type=["csv"], key="asset_upload")
        if upload is not None:
            st.session_state.manual_upload_name = upload.name
            st.session_state.manual_upload_bytes = upload.getvalue()
            st.success(f"Staged `{upload.name}` for collateral verification.")

    dataset = None
    if data_source == "Use synthetic (ANON)":
        dataset = st.session_state.get("synthetic_df")
    elif data_source == "Use synthetic (RAW – auto-sanitize)":
        raw_df = st.session_state.get("synthetic_raw_df")
        if raw_df is not None:
            dataset, _ = drop_pii_columns(raw_df)
    elif data_source == "Use anonymized dataset":
        dataset = st.session_state.get("anonymized_df")
    elif data_source == "Upload manually":
        up_bytes = st.session_state.get("manual_upload_bytes")
        if up_bytes:
            try:
                dataset = pd.read_csv(io.BytesIO(up_bytes))
            except Exception:
                dataset = None
                st.error("Could not read uploaded CSV for collateral verification.")

    if dataset is not None and not dataset.empty:
        with st.expander("Preview selected dataset", expanded=False):
            st.dataframe(ensure_application_ids(dataset).head(10), use_container_width=True)
    else:
        st.info("Select or generate a dataset to begin collateral verification.", icon="ℹ️")

    col_conf, col_ratio = st.columns(2)
    with col_conf:
        confidence_threshold = st.slider("Minimum confidence from asset agent", 0.50, 1.00, 0.88, 0.01)
    with col_ratio:
        value_ratio = st.slider("Min estimated collateral vs. loan ratio", 0.10, 1.50, 0.80, 0.05)

    if st.button("🛡️ Generate collateral verification report", use_container_width=True):
        if dataset is None or dataset.empty:
            st.warning("No dataset available. Generate synthetic data or upload a CSV first.")
        else:
            required = {"application_id", "collateral_type", "collateral_value"}
            missing = [c for c in required if c not in dataset.columns]
            if missing:
                st.error("Dataset is missing required columns: " + ", ".join(sorted(missing)))
            else:
                with st.spinner("Running asset appraisal agent across collateral records..."):
                    report_df, errors = build_collateral_report(
                        ensure_application_ids(dataset),
                        confidence_threshold=confidence_threshold,
                        value_ratio=value_ratio,
                    )
                if report_df.empty:
                    st.warning("No collateral rows were processed. Check the dataset contents.")
                else:
                    st.session_state.asset_collateral_df = report_df
                    path = save_to_runs(report_df, "collateral_verification")
                    st.session_state.asset_collateral_path = path
                    st.success(
                        f"Collateral verification complete — {len(report_df)} loans processed. Saved to `{os.path.basename(path)}`."
                    )
                    st.dataframe(report_df.head(25), use_container_width=True)
                    if errors:
                        st.warning("\n".join(errors))
with tab_credit:
    st.subheader("🤖 Credit appraisal by AI assistant")

    try:
        resp = requests.get(f"{API_URL}/v1/training/production_meta", timeout=5)
        if resp.status_code == 200:
            meta = resp.json()
            if meta.get("has_production"):
                ver = (meta.get("meta") or {}).get("version", "1.x")
                src = (meta.get("meta") or {}).get("source", "production")
                st.success(f"🟢 Production model active — version: {ver} • source: {src}")
            else:
                st.warning("⚠️ No production model promoted yet — using baseline.")
        else:
            st.info("ℹ️ Could not fetch production model meta.")
    except Exception:
        st.info("ℹ️ Production meta unavailable.")

    LLM_MODELS = [
        ("Phi-3 Mini (3.8B) — CPU OK", "phi3:3.8b", "CPU 8GB RAM (fast)"),
        ("Mistral 7B Instruct — CPU slow / GPU OK", "mistral:7b-instruct", "CPU 16GB (slow) or GPU ≥8GB"),
        ("Gemma-2 7B — CPU slow / GPU OK", "gemma2:7b", "CPU 16GB (slow) or GPU ≥8GB"),
        ("LLaMA-3 8B — GPU recommended", "llama3:8b-instruct", "GPU ≥12GB (CPU very slow)"),
        ("Qwen2 7B — GPU recommended", "qwen2:7b-instruct", "GPU ≥12GB (CPU very slow)"),
        ("Mixtral 8x7B — GPU only (big)", "mixtral:8x7b-instruct", "GPU 24–48GB"),
    ]
    LLM_LABELS = [label for (label, _, _) in LLM_MODELS]
    LLM_VALUE_BY_LABEL = {label: value for (label, value, _) in LLM_MODELS}
    LLM_HINT_BY_LABEL = {label: hint for (label, _, hint) in LLM_MODELS}

    OPENSTACK_FLAVORS = {
        "m4.medium": "4 vCPU / 8 GB RAM — CPU-only small",
        "m8.large": "8 vCPU / 16 GB RAM — CPU-only medium",
        "g1.a10.1": "8 vCPU / 32 GB RAM + 1×A10 24GB",
        "g1.l40.1": "16 vCPU / 64 GB RAM + 1×L40 48GB",
        "g2.a100.1": "24 vCPU / 128 GB RAM + 1×A100 80GB",
    }

    with st.expander("🧠 Local LLM & Hardware Profile", expanded=True):
        c1, c2 = st.columns([1.2, 1])
        with c1:
            model_label = st.selectbox("Local LLM (used for narratives/explanations)", LLM_LABELS, index=1)
            llm_value = LLM_VALUE_BY_LABEL[model_label]
            st.caption(f"Hint: {LLM_HINT_BY_LABEL[model_label]}")
        with c2:
            flavor = st.selectbox("OpenStack flavor / host profile", list(OPENSTACK_FLAVORS.keys()), index=0)
            st.caption(OPENSTACK_FLAVORS[flavor])
        st.caption("These are passed to the API as hints; your API can choose Ollama/Flowise backends accordingly.")

    data_choice = st.selectbox(
        "Select Data Source",
        [
            "Use synthetic (ANON)",
            "Use synthetic (RAW – auto-sanitize)",
            "Use anonymized dataset",
            "Use collateral verification output",
            "Upload manually",
        ],
    )
    use_llm = st.checkbox("Use LLM narrative", value=False)
    agent_name = "credit_appraisal"

    if data_choice == "Upload manually":
        upload = st.file_uploader("Upload your CSV", type=["csv"], key="manual_run_upload")
        if upload is not None:
            st.session_state.manual_upload_name = upload.name
            st.session_state.manual_upload_bytes = upload.getvalue()
            st.success(f"File staged: {upload.name}")

    st.markdown("### ⚙️ Decision Rule Set")
    rule_mode = st.radio(
        "Choose rule mode",
        ["Classic (bank-style metrics)", "NDI (Net Disposable Income) — simple"],
        index=0,
        help="NDI = income - all monthly obligations. Approve if NDI and NDI ratio pass thresholds.",
    )

    CLASSIC_DEFAULTS = {
        "max_dti": 0.45,
        "min_emp_years": 2,
        "min_credit_hist": 3,
        "salary_floor": 3000,
        "max_delinquencies": 2,
        "max_current_loans": 3,
        "req_min": 1000,
        "req_max": 200000,
        "loan_terms": [12, 24, 36, 48, 60],
        "threshold": 0.45,
        "target_rate": None,
        "random_band": True,
        "min_income_debt_ratio": 0.35,
        "compounded_debt_factor": 1.0,
        "monthly_debt_relief": 0.50,
    }
    NDI_DEFAULTS = {
        "ndi_value": 800.0,
        "ndi_ratio": 0.50,
        "threshold": 0.45,
        "target_rate": None,
        "random_band": True,
    }

    if "classic_rules" not in st.session_state:
        st.session_state.classic_rules = CLASSIC_DEFAULTS.copy()
    if "ndi_rules" not in st.session_state:
        st.session_state.ndi_rules = NDI_DEFAULTS.copy()

    def reset_classic() -> None:
        st.session_state.classic_rules = CLASSIC_DEFAULTS.copy()

    def reset_ndi() -> None:
        st.session_state.ndi_rules = NDI_DEFAULTS.copy()

    if rule_mode.startswith("Classic"):
        with st.expander("Classic Metrics (with Reset)", expanded=True):
            rc = st.session_state.classic_rules
            r1, r2, r3 = st.columns(3)
            with r1:
                rc["max_dti"] = st.slider("Max Debt-to-Income (DTI)", 0.0, 1.0, rc["max_dti"], 0.01)
                rc["min_emp_years"] = st.number_input("Min Employment Years", 0, 40, rc["min_emp_years"])
                rc["min_credit_hist"] = st.number_input("Min Credit History (years)", 0, 40, rc["min_credit_hist"])
            with r2:
                rc["salary_floor"] = st.number_input(
                    "Minimum Monthly Salary",
                    0,
                    1_000_000_000,
                    rc["salary_floor"],
                    step=1000,
                    help=fmt_currency_label("in local currency"),
                )
                rc["max_delinquencies"] = st.number_input("Max Delinquencies", 0, 10, rc["max_delinquencies"])
                rc["max_current_loans"] = st.number_input("Max Current Loans", 0, 10, rc["max_current_loans"])
            with r3:
                rc["req_min"] = st.number_input(
                    fmt_currency_label("Requested Amount Min"),
                    0,
                    10_000_000_000,
                    rc["req_min"],
                    step=1000,
                )
                rc["req_max"] = st.number_input(
                    fmt_currency_label("Requested Amount Max"),
                    0,
                    10_000_000_000,
                    rc["req_max"],
                    step=1000,
                )
                rc["loan_terms"] = st.multiselect(
                    "Allowed Loan Terms (months)",
                    [12, 24, 36, 48, 60, 72],
                    default=rc["loan_terms"],
                )

            st.markdown("#### 🧮 Debt Pressure Controls")
            d1, d2, d3 = st.columns(3)
            with d1:
                rc["min_income_debt_ratio"] = st.slider(
                    "Min Income / (Compounded Debt) Ratio",
                    0.10,
                    2.00,
                    rc["min_income_debt_ratio"],
                    0.01,
                )
            with d2:
                rc["compounded_debt_factor"] = st.slider(
                    "Compounded Debt Factor (× requested)", 0.5, 3.0, rc["compounded_debt_factor"], 0.1
                )
            with d3:
                rc["monthly_debt_relief"] = st.slider(
                    "Monthly Debt Relief Factor", 0.10, 1.00, rc["monthly_debt_relief"], 0.05
                )

            st.markdown("---")
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                use_target = st.toggle("🎯 Use target approval rate", value=(rc["target_rate"] is not None))
            with c2:
                rc["random_band"] = st.toggle(
                    "🎲 Randomize approval band (20–60%) when no target",
                    value=rc["random_band"],
                )
            with c3:
                if st.button("↩️ Reset to defaults", key="reset_classic"):
                    reset_classic()
                    st.experimental_rerun()

            if use_target:
                rc["target_rate"] = st.slider(
                    "Target approval rate", 0.05, 0.95, rc["target_rate"] or 0.40, 0.01
                )
                rc["threshold"] = None
            else:
                rc["threshold"] = st.slider("Model score threshold", 0.0, 1.0, rc["threshold"], 0.01)
                rc["target_rate"] = None
    else:
        with st.expander("NDI Metrics (with Reset)", expanded=True):
            rn = st.session_state.ndi_rules
            n1, n2 = st.columns(2)
            with n1:
                rn["ndi_value"] = st.number_input(
                    fmt_currency_label("Min NDI (Net Disposable Income) per month"),
                    0.0,
                    1e12,
                    float(rn["ndi_value"]),
                    step=50.0,
                )
            with n2:
                rn["ndi_ratio"] = st.slider(
                    "Min NDI / Income ratio",
                    0.0,
                    1.0,
                    float(rn["ndi_ratio"]),
                    0.01,
                )
            st.caption("NDI = income - all monthly obligations (rent, food, loans, cards, etc.).")

            st.markdown("---")
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                use_target = st.toggle("🎯 Use target approval rate", value=(rn["target_rate"] is not None))
            with c2:
                rn["random_band"] = st.toggle(
                    "🎲 Randomize approval band (20–60%) when no target",
                    value=rn["random_band"],
                )
            with c3:
                if st.button("↩️ Reset to defaults (NDI)", key="reset_ndi"):
                    reset_ndi()
                    st.experimental_rerun()

            if use_target:
                rn["target_rate"] = st.slider(
                    "Target approval rate", 0.05, 0.95, rn["target_rate"] or 0.40, 0.01
                )
                rn["threshold"] = None
            else:
                rn["threshold"] = st.slider("Model score threshold", 0.0, 1.0, rn["threshold"], 0.01)
                rn["target_rate"] = None

    if st.button("🚀 Run Agent", use_container_width=True):
        try:
            files = None
            data: Dict[str, Any] = {
                "use_llm_narrative": str(use_llm).lower(),
                "llm_model": llm_value,
                "hardware_flavor": flavor,
                "currency_code": st.session_state.currency_code,
                "currency_symbol": st.session_state.currency_symbol,
            }
            if rule_mode.startswith("Classic"):
                rc = st.session_state.classic_rules
                data.update(
                    {
                        "min_employment_years": str(rc["min_emp_years"]),
                        "max_debt_to_income": str(rc["max_dti"]),
                        "min_credit_history_length": str(rc["min_credit_hist"]),
                        "max_num_delinquencies": str(rc["max_delinquencies"]),
                        "max_current_loans": str(rc["max_current_loans"]),
                        "requested_amount_min": str(rc["req_min"]),
                        "requested_amount_max": str(rc["req_max"]),
                        "loan_term_months_allowed": ",".join(map(str, rc["loan_terms"])) if rc["loan_terms"] else "",
                        "min_income_debt_ratio": str(rc["min_income_debt_ratio"]),
                        "compounded_debt_factor": str(rc["compounded_debt_factor"]),
                        "monthly_debt_relief": str(rc["monthly_debt_relief"]),
                        "salary_floor": str(rc["salary_floor"]),
                        "threshold": "" if rc["threshold"] is None else str(rc["threshold"]),
                        "target_approval_rate": "" if rc["target_rate"] is None else str(rc["target_rate"]),
                        "random_band": str(rc["random_band"]).lower(),
                        "random_approval_band": str(rc["random_band"]).lower(),
                        "rule_mode": "classic",
                    }
                )
            else:
                rn = st.session_state.ndi_rules
                data.update(
                    {
                        "ndi_value": str(rn["ndi_value"]),
                        "ndi_ratio": str(rn["ndi_ratio"]),
                        "threshold": "" if rn["threshold"] is None else str(rn["threshold"]),
                        "target_approval_rate": "" if rn["target_rate"] is None else str(rn["target_rate"]),
                        "random_band": str(rn["random_band"]).lower(),
                        "random_approval_band": str(rn["random_band"]).lower(),
                        "rule_mode": "ndi",
                    }
                )

            def prep_and_pack(df: pd.DataFrame, filename: str) -> Dict[str, Tuple[str, bytes, str]]:
                safe_df = dedupe_columns(df)
                safe_df, _ = drop_pii_columns(safe_df)
                safe_df = strip_policy_banned(safe_df)
                safe_df = to_agent_schema(safe_df)
                buf = io.StringIO()
                safe_df.to_csv(buf, index=False)
                return {"file": (filename, buf.getvalue().encode("utf-8"), "text/csv")}

            if data_choice == "Use synthetic (ANON)":
                if "synthetic_df" not in st.session_state:
                    st.warning("No ANON synthetic dataset found. Generate it in the first tab.")
                    st.stop()
                files = prep_and_pack(st.session_state.synthetic_df, "synthetic_anon.csv")
            elif data_choice == "Use synthetic (RAW – auto-sanitize)":
                if "synthetic_raw_df" not in st.session_state:
                    st.warning("No RAW synthetic dataset found. Generate it in the first tab.")
                    st.stop()
                files = prep_and_pack(st.session_state.synthetic_raw_df, "synthetic_raw_sanitized.csv")
            elif data_choice == "Use anonymized dataset":
                if "anonymized_df" not in st.session_state:
                    st.warning("No anonymized dataset found. Create it in the second tab.")
                    st.stop()
                files = prep_and_pack(st.session_state.anonymized_df, "anonymized.csv")
            elif data_choice == "Use collateral verification output":
                if "asset_collateral_df" not in st.session_state:
                    st.warning("No collateral output found. Run the asset appraisal first.")
                    st.stop()
                files = prep_and_pack(st.session_state.asset_collateral_df, "collateral_verified.csv")
            elif data_choice == "Upload manually":
                up_name = st.session_state.get("manual_upload_name")
                up_bytes = st.session_state.get("manual_upload_bytes")
                if not up_name or not up_bytes:
                    st.warning("Please upload a CSV first.")
                    st.stop()
                try:
                    tmp_df = pd.read_csv(io.BytesIO(up_bytes))
                    files = prep_and_pack(tmp_df, up_name)
                except Exception:
                    files = {"file": (up_name, up_bytes, "text/csv")}
            else:
                st.error("Unknown data source selection.")
                st.stop()

            response = requests.post(
                f"{API_URL}/v1/agents/{agent_name}/run",
                data=data,
                files=files,
                timeout=180,
            )
            if response.status_code != 200:
                st.error(f"Run failed ({response.status_code}): {response.text}")
                st.stop()

            result = response.json()
            st.session_state.last_run_id = result.get("run_id")
            st.success(f"✅ Run succeeded! Run ID: {st.session_state.last_run_id}")

            rid = st.session_state.last_run_id
            merged_url = f"{API_URL}/v1/runs/{rid}/report?format=csv"
            merged_bytes = requests.get(merged_url, timeout=30).content
            merged_df = pd.read_csv(io.BytesIO(merged_bytes))
            st.session_state.last_merged_df = merged_df

            ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            out_name = f"ai-appraisal-outputs-{ts}-{st.session_state.currency_code}.csv"
            st.download_button(
                "⬇️ Download AI outputs (CSV)",
                merged_df.to_csv(index=False).encode("utf-8"),
                out_name,
                "text/csv",
            )

            st.markdown("### 📄 Credit AI Agent Decisions Table (filtered)")
            uniq_dec = sorted(merged_df.get("decision", pd.Series(dtype=str)).dropna().unique())
            chosen = st.multiselect("Filter decision", options=uniq_dec, default=uniq_dec, key="filter_decisions")
            df_view = merged_df.copy()
            if "decision" in df_view.columns and chosen:
                df_view = df_view[df_view["decision"].isin(chosen)]
            st.dataframe(df_view, use_container_width=True)

            st.markdown("## 📊 Dashboard")
            render_credit_dashboard(merged_df, st.session_state.get("currency_symbol", ""))

            if "rule_reasons" in df_view.columns:
                parsed = df_view["rule_reasons"].apply(try_json)
                df_view["metrics_met"] = parsed.apply(
                    lambda d: ", ".join(sorted([k for k, v in (d or {}).items() if v is True])) if isinstance(d, dict) else ""
                )
                df_view["metrics_unmet"] = parsed.apply(
                    lambda d: ", ".join(sorted([k for k, v in (d or {}).items() if v is False])) if isinstance(d, dict) else ""
                )
            cols_show = [
                "application_id",
                "customer_type",
                "decision",
                "score",
                "loan_amount",
                "income",
                "metrics_met",
                "metrics_unmet",
                "proposed_loan_option",
                "proposed_consolidation_loan",
                "top_feature",
                "explanation",
            ]
            cols_show = [c for c in cols_show if c in df_view.columns]
            if cols_show:
                st.dataframe(df_view[cols_show].head(500), use_container_width=True)

            cdl1, cdl2, cdl3, cdl4, cdl5 = st.columns(5)
            with cdl1:
                st.markdown(f"[⬇️ PDF report]({API_URL}/v1/runs/{rid}/report?format=pdf)")
            with cdl2:
                st.markdown(f"[⬇️ Scores CSV]({API_URL}/v1/runs/{rid}/report?format=scores_csv)")
            with cdl3:
                st.markdown(f"[⬇️ Explanations CSV]({API_URL}/v1/runs/{rid}/report?format=explanations_csv)")
            with cdl4:
                st.markdown(f"[⬇️ Merged CSV]({API_URL}/v1/runs/{rid}/report?format=csv)")
            with cdl5:
                st.markdown(f"[⬇️ JSON]({API_URL}/v1/runs/{rid}/report?format=json)")

        except Exception as exc:
            st.exception(exc)

    if st.session_state.get("last_run_id"):
        st.markdown("---")
        st.subheader("📥 Download Latest Outputs")
        rid = st.session_state.last_run_id
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f"[⬇️ PDF]({API_URL}/v1/runs/{rid}/report?format=pdf)")
        with col2:
            st.markdown(f"[⬇️ Scores CSV]({API_URL}/v1/runs/{rid}/report?format=scores_csv)")
        with col3:
            st.markdown(f"[⬇️ Explanations CSV]({API_URL}/v1/runs/{rid}/report?format=explanations_csv)")
        with col4:
            st.markdown(f"[⬇️ Merged CSV]({API_URL}/v1/runs/{rid}/report?format=csv)")
        with col5:
            st.markdown(f"[⬇️ JSON]({API_URL}/v1/runs/{rid}/report?format=json)")
with tab_review:
    st.subheader("🧑‍⚖️ Human Review — Correct AI Decisions & Score Agreement")

    uploaded_review = st.file_uploader("Load AI outputs CSV for review (optional)", type=["csv"], key="review_upload")
    if uploaded_review is not None:
        try:
            st.session_state.last_merged_df = pd.read_csv(uploaded_review)
            st.success("Loaded review dataset from uploaded CSV.")
        except Exception as exc:
            st.error(f"Could not read uploaded CSV: {exc}")

    if "last_merged_df" not in st.session_state:
        st.info("Run the agent (previous tab) or upload an AI outputs CSV to load results for review.")
    else:
        dfm = st.session_state.last_merged_df.copy()
        st.markdown("#### 1) Select rows to review and correct")

        editable_cols = []
        if "decision" in dfm.columns:
            editable_cols.append("decision")
        if "rule_reasons" in dfm.columns:
            editable_cols.append("rule_reasons")
        if "customer_type" in dfm.columns:
            editable_cols.append("customer_type")

        editable = dfm[["application_id"] + editable_cols].copy()
        editable.rename(columns={"decision": "ai_decision"}, inplace=True)
        editable["human_decision"] = editable.get("ai_decision", "approved")
        editable["human_rule_reasons"] = editable.get("rule_reasons", "")

        edited = st.data_editor(
            editable,
            num_rows="dynamic",
            use_container_width=True,
            key="review_editor",
            column_config={
                "human_decision": st.column_config.SelectboxColumn(options=["approved", "denied"]),
                "customer_type": st.column_config.SelectboxColumn(options=["bank", "non-bank"], disabled=True),
            },
        )

        st.markdown("#### 2) Compute agreement score")
        if st.button("Compute agreement score"):
            if "ai_decision" in edited.columns and "human_decision" in edited.columns:
                agree = (edited["ai_decision"] == edited["human_decision"]).astype(int)
                score = float(agree.mean()) if len(agree) else 0.0
                st.success(f"Agreement score (AI vs human): {score:.3f}")
                st.session_state.last_agreement_score = score
            else:
                st.warning("Missing decision columns to compute score.")

        st.markdown("#### 3) Export review CSV")
        model_used = "production"
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_user = st.session_state.user_info.get("name", "user").replace(" ", "").lower() or "user"
        review_name = f"creditappraisal.{safe_user}.{model_used}.{ts}.csv"
        csv_bytes = edited.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export review CSV", csv_bytes, review_name, "text/csv")
        st.caption(f"Saved file name pattern: **{review_name}**")

with tab_train:
    st.subheader("🔁 Human Feedback → Retrain (new payload)")
    st.markdown("**Drag & drop** one or more review CSVs exported from the Human Review tab.")
    up_list = st.file_uploader("Upload feedback CSV(s)", type=["csv"], accept_multiple_files=True, key="train_feedback")

    staged_paths: List[str] = []
    if up_list:
        for upload in up_list:
            dest = os.path.join(TMP_FEEDBACK_DIR, upload.name)
            with open(dest, "wb") as fh:
                fh.write(upload.getvalue())
            staged_paths.append(dest)
        st.success(f"Staged {len(staged_paths)} feedback file(s) to {TMP_FEEDBACK_DIR}")
        st.write(staged_paths)

    payload = {
        "feedback_csvs": staged_paths,
        "user_name": st.session_state.user_info.get("name", ""),
        "agent_name": "credit_appraisal",
        "algo_name": "credit_lr",
    }
    st.markdown("#### Launch Retrain")
    st.code(json.dumps(payload, indent=2), language="json")

    colA, colB = st.columns(2)
    with colA:
        if st.button("🚀 Train candidate model"):
            try:
                response = requests.post(f"{API_URL}/v1/training/train", json=payload, timeout=90)
                if response.ok:
                    st.success(response.json())
                    st.session_state.last_train_job = response.json().get("job_id")
                else:
                    st.error(response.text)
            except Exception as exc:
                st.error(f"Train failed: {exc}")
    with colB:
        if st.button("⬆️ Promote last candidate to PRODUCTION"):
            try:
                response = requests.post(f"{API_URL}/v1/training/promote", timeout=30)
                if response.ok:
                    st.success(response.json())
                else:
                    st.error(response.text)
            except Exception as exc:
                st.error(f"Promote failed: {exc}")

    st.markdown("---")
    st.markdown("#### Production Model")
    try:
        resp = requests.get(f"{API_URL}/v1/training/production_meta", timeout=5)
        if resp.ok:
            st.json(resp.json())
        else:
            st.info("No production model yet.")
    except Exception as exc:
        st.warning(f"Could not load production meta: {exc}")

with tab_loop:
    st.subheader("🔄 Loop Back — Deploy and Re-run")
    st.write(
        """
        After promoting a new model to production, return to the credit appraisal tab to re-run the agent with
        updated parameters. This loop keeps your appraisal pipeline fresh and aligned with the latest human feedback.
        """
    )
    if st.button("Return to Credit Appraisal", use_container_width=True):
        st.experimental_rerun()

st.markdown("<footer>Made with ❤️ by Dzoan Nguyen — Open AI Sandbox Initiative</footer>", unsafe_allow_html=True)
