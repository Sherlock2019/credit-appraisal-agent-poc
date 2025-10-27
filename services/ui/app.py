from __future__ import annotations

import datetime
import io
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# ────────────────────────────────
# PAGE CONFIG
# ────────────────────────────────
st.set_page_config(
    page_title="AI Agent Sandbox — By the People, For the People",
    layout="wide",
)

# ────────────────────────────────
# CONSTANTS / PATHS
# ────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8090")
RUNS_DIR = os.path.expanduser("~/credit-appraisal-agent-poc/services/api/.runs")
TMP_FEEDBACK_DIR = os.path.join(RUNS_DIR, "tmp_feedback")
LANDING_IMG_DIR = os.path.expanduser("~/credit-appraisal-agent-poc/services/ui/landing_images")

os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(TMP_FEEDBACK_DIR, exist_ok=True)
os.makedirs(LANDING_IMG_DIR, exist_ok=True)


def clear_query_params() -> None:
    """Clear query parameters across Streamlit versions."""
    try:
        st.query_params.clear()
    except Exception:
        try:
            st.experimental_set_query_params()
        except Exception:
            pass


def set_stage(stage: str, *, update_query: bool = True) -> None:
    """Centralised helper to update the stage and optional query params."""
    st.session_state["stage"] = stage
    if not update_query:
        return
    try:
        st.query_params.from_dict({"stage": stage})
    except Exception:
        try:
            st.experimental_set_query_params(stage=stage)
        except Exception:
            pass

CURRENCY_OPTIONS: Dict[str, Tuple[str, str, float]] = {
    "VND (₫)": ("VND", "₫", 1.0),
    "USD ($)": ("USD", "$", 1.0 / 24000.0),
    "EUR (€)": ("EUR", "€", 1.0 / 26000.0),
}
DEFAULT_CURRENCY = "VND (₫)"

PIPELINE_STAGES: List[Tuple[str, str, str]] = [
    (
        "data",
        "🏦 Synthetic Data Generator",
        "Create localized loan books and seed downstream stages with consistent datasets.",
    ),
    (
        "kyc",
        "🛂 KYC Agent",
        "Prepare compliance dossiers, sanctions screening, and risk signals before underwriting.",
    ),
    (
        "asset",
        "🏛️ Asset Appraisal AI Agent",
        "Verify collateral valuations, appraisal confidence, and eligibility for credit decisions.",
    ),
    (
        "credit",
        "🤖 Credit Appraisal AI Assistant",
        "Score applications with explainable AI narratives and policy-aligned decisioning.",
    ),
    (
        "review",
        "🧑‍⚖️ Human Review",
        "Audit AI outputs, adjust verdicts, and capture agreement plus remediation notes.",
    ),
    (
        "training",
        "🔁 Training (Feedback → Retrain)",
        "Loop curated feedback into retraining jobs and promote production-ready models.",
    ),
    (
        "loopback",
        "🔄 Loop Back to Step 3",
        "Review promoted models and relaunch the credit appraisal agent with the latest settings.",
    ),
]

AGENTS = [
    (
        "🏦 Banking & Finance",
        "💰 Retail Banking",
        "💳 Credit Appraisal Agent",
        "Explainable AI for loan decisioning",
        "Available",
        "💳",
    ),
    (
        "🏦 Banking & Finance",
        "💰 Retail Banking",
        "🏦 Asset Appraisal Agent",
        "Market-driven collateral valuation",
        "Coming Soon",
        "🏦",
    ),
    (
        "🏦 Banking & Finance",
        "🩺 Insurance",
        "🩺 Claims Triage Agent",
        "Automated claims prioritization",
        "Coming Soon",
        "🩺",
    ),
    (
        "⚡ Energy & Sustainability",
        "🔋 EV & Charging",
        "⚡ EV Charger Optimizer",
        "Optimize charger deployment via AI",
        "Coming Soon",
        "⚡",
    ),
    (
        "⚡ Energy & Sustainability",
        "☀️ Solar",
        "☀️ Solar Yield Estimator",
        "Estimate solar ROI and efficiency",
        "Coming Soon",
        "☀️",
    ),
    (
        "🚗 Automobile & Transport",
        "🚙 Automobile",
        "🚗 Predictive Maintenance",
        "Prevent downtime via sensor analytics",
        "Coming Soon",
        "🚗",
    ),
    (
        "🚗 Automobile & Transport",
        "🔋 EV",
        "🔋 EV Battery Health Agent",
        "Monitor EV battery health cycles",
        "Coming Soon",
        "🔋",
    ),
    (
        "🚗 Automobile & Transport",
        "🚚 Ride-hailing / Logistics",
        "🛻 Fleet Route Optimizer",
        "Dynamic route optimization for fleets",
        "Coming Soon",
        "🛻",
    ),
    (
        "💻 Information Technology",
        "🧰 Support & Security",
        "🧩 IT Ticket Triage",
        "Auto-prioritize support tickets",
        "Coming Soon",
        "🧩",
    ),
    (
        "💻 Information Technology",
        "🛡️ Security",
        "🔐 SecOps Log Triage",
        "Detect anomalies & summarize alerts",
        "Coming Soon",
        "🔐",
    ),
    (
        "⚖️ Legal & Government",
        "⚖️ Law Firms",
        "⚖️ Contract Analyzer",
        "Extract clauses and compliance risks",
        "Coming Soon",
        "⚖️",
    ),
    (
        "⚖️ Legal & Government",
        "🏛️ Public Services",
        "🏛️ Citizen Service Agent",
        "Smart assistant for citizen services",
        "Coming Soon",
        "🏛️",
    ),
    (
        "🛍️ Retail / SMB / Creative",
        "🏬 Retail & eCommerce",
        "📈 Sales Forecast Agent",
        "Predict demand & inventory trends",
        "Coming Soon",
        "📈",
    ),
    (
        "🎬 Retail / SMB / Creative",
        "🎨 Media & Film",
        "🎬 Budget Cost Assistant",
        "Estimate, optimize, and track film & production costs using AI",
        "Coming Soon",
        "🎬",
    ),
]

# ────────────────────────────────
# SESSION DEFAULTS
# ────────────────────────────────
st.session_state.setdefault("stage", "landing")
st.session_state.setdefault("logged_in", False)
st.session_state.setdefault("user_info", {"name": "", "email": "", "flagged": False})
st.session_state.setdefault("workflow_stage", "data")
st.session_state.setdefault("currency_code_label", DEFAULT_CURRENCY)

try:
    query_params = st.query_params
except Exception:
    query_params = {}

target_stage = query_params.get("stage") if isinstance(query_params, dict) else None
if isinstance(target_stage, list):
    target_stage = target_stage[0] if target_stage else None

if target_stage in {"landing", "agents", "login", "credit_agent"} and target_stage != st.session_state["stage"]:
    set_stage(target_stage, update_query=False)
    clear_query_params()
    st.rerun()

if query_params.get("agent") == "credit" and st.session_state["stage"] != "login":
    set_stage("login", update_query=False)
    clear_query_params()
    st.rerun()


# ────────────────────────────────
# STYLE BLOCKS
# ────────────────────────────────
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
    .pipeline-hero {
        background: linear-gradient(135deg, rgba(37,99,235,0.25), rgba(59,130,246,0.55));
        border-radius: 28px;
        padding: 32px 36px;
        margin-bottom: 32px;
        border: 1px solid rgba(148, 163, 184, 0.2);
        box-shadow: 0 20px 40px rgba(15, 23, 42, 0.45);
    }
    .pipeline-hero__eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 0.75rem;
        color: rgba(226, 232, 240, 0.85);
    }
    .pipeline-hero__header h1 {
        font-size: 3rem;
        font-weight: 900;
        margin: 12px 0 8px;
        color: #ffffff;
    }
    .pipeline-hero__header p {
        font-size: 1.1rem;
        color: #e2e8f0;
        max-width: 720px;
        margin-bottom: 0;
    }
    .pipeline-steps {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 18px;
        margin-top: 26px;
    }
    .pipeline-step {
        background: rgba(15, 23, 42, 0.75);
        border-radius: 20px;
        padding: 20px 22px;
        border: 1px solid rgba(148, 163, 184, 0.18);
        display: flex;
        gap: 16px;
        align-items: flex-start;
        min-height: 140px;
    }
    .pipeline-step--active {
        background: linear-gradient(160deg, rgba(59, 130, 246, 0.28), rgba(37, 99, 235, 0.42));
        border-color: rgba(96, 165, 250, 0.65);
        box-shadow: 0 18px 32px rgba(30, 64, 175, 0.35);
    }
    .pipeline-step__index {
        width: 42px;
        height: 42px;
        border-radius: 14px;
        background: linear-gradient(180deg,#38bdf8,#2563eb);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        color: #0f172a;
        font-size: 1.1rem;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.35);
    }
    .pipeline-step__index--active {
        background: linear-gradient(180deg,#facc15,#f59e0b);
        color: #1f2937;
    }
    .pipeline-step__body { flex: 1; }
    .pipeline-step__title {
        font-weight: 700;
        font-size: 1.1rem;
        color: #f8fafc;
        margin-bottom: 6px;
    }
    .pipeline-step__body p {
        margin: 0;
        font-size: 0.95rem;
        color: rgba(226,232,240,0.85);
    }
    footer {
        text-align: center;
        padding: 2rem;
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────
# HELPERS
# ────────────────────────────────

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


def set_currency_defaults() -> None:
    code_label = st.session_state.get("currency_code_label", DEFAULT_CURRENCY)
    code, sym, fx = CURRENCY_OPTIONS.get(code_label, CURRENCY_OPTIONS[DEFAULT_CURRENCY])
    st.session_state["currency_code_label"] = code_label
    st.session_state["currency_code"] = code
    st.session_state["currency_symbol"] = sym
    st.session_state["currency_fx"] = fx


def fmt_currency_label(text: str) -> str:
    sym = st.session_state.get("currency_symbol", "")
    return f"{text} ({sym})" if sym else text


def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols: List[str] = []
    seen: Dict[str, int] = {}
    for column in df.columns:
        if column in seen:
            seen[column] += 1
            cols.append(f"{column}.{seen[column]}")
        else:
            seen[column] = 0
            cols.append(column)
    out = df.copy()
    out.columns = cols
    return out


def drop_pii_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    pii_like = {"name", "email", "phone", "ssn", "address", "dob", "national_id"}
    to_drop = [c for c in df.columns if any(flag in c.lower() for flag in pii_like)]
    out = df.drop(columns=to_drop, errors="ignore")
    return dedupe_columns(out), to_drop


def strip_policy_banned(df: pd.DataFrame) -> pd.DataFrame:
    return df


def to_agent_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "application_id" not in out.columns:
        out["application_id"] = [f"APP_{i:04d}" for i in range(1, len(out) + 1)]
    out["application_id"] = out["application_id"].astype(str)
    return out


def _to_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        return default


def try_json(obj: Any) -> Optional[Dict[str, Any]]:
    if isinstance(obj, dict):
        return obj
    if not isinstance(obj, str):
        return None
    try:
        return json.loads(obj)
    except Exception:
        return None


def ensure_application_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "application_id" not in out.columns:
        out["application_id"] = [f"APP_{i:04d}" for i in range(1, len(out) + 1)]
    out["application_id"] = out["application_id"].astype(str)
    return out


def save_to_runs(df: pd.DataFrame, prefix: str) -> str:
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    flagged = bool(st.session_state.get("user_info", {}).get("flagged", False))
    flag_suffix = "_FLAGGED" if flagged else ""
    fname = f"{prefix}_{ts}{flag_suffix}.csv"
    fpath = os.path.join(RUNS_DIR, fname)
    dedupe_columns(df).to_csv(fpath, index=False)
    return fpath


def logout_user() -> None:
    keys_to_clear = [
        "synthetic_raw_df",
        "synthetic_df",
        "anonymized_df",
        "asset_collateral_df",
        "asset_collateral_path",
        "manual_upload_name",
        "manual_upload_bytes",
        "last_merged_df",
        "review_corrections",
        "last_run_id",
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)
    st.session_state.logged_in = False
    st.session_state.user_info = {"name": "", "email": "", "flagged": False}
    st.session_state.workflow_stage = "data"
    set_stage("landing")
    clear_query_params()
    st.rerun()


set_currency_defaults()

# ────────────────────────────────
# DATA GENERATION UTILITIES
# ────────────────────────────────

def generate_raw_synthetic(n: int = 200, non_bank_ratio: float = 0.30) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    ids = [f"APP_{i:04d}" for i in range(1, n + 1)]
    income = rng.integers(5_000_000, 90_000_000, size=n)
    loan_amount = (income * rng.uniform(0.5, 3.0, size=n)).astype(int)
    collateral_value = (loan_amount * rng.uniform(0.6, 1.5, size=n)).astype(int)
    customer_type = rng.choice(["Bank", "Non-bank"], p=[1 - non_bank_ratio, non_bank_ratio], size=n)
    df = pd.DataFrame(
        {
            "application_id": ids,
            "customer_name": [f"Name{i}" for i in range(n)],
            "email": [f"user{i}@mail.local" for i in range(n)],
            "phone": [f"+84-09{rng.integers(1000000, 9999999)}" for _ in range(n)],
            "income": income,
            "loan_amount": loan_amount,
            "collateral_type": rng.choice(["House", "Car", "Gold", "Land"], size=n),
            "collateral_value": collateral_value,
            "customer_type": customer_type,
            "currency_code": st.session_state.get("currency_code", "VND"),
        }
    )
    fx = st.session_state.get("currency_fx", 1.0)
    if fx != 1.0:
        for column in ("income", "loan_amount", "collateral_value"):
            if column in df.columns:
                df[column] = (df[column] * fx).round(2)
    return df


def generate_anon_synthetic(n: int = 200, non_bank_ratio: float = 0.30) -> pd.DataFrame:
    raw = generate_raw_synthetic(n, non_bank_ratio)
    out, _ = drop_pii_columns(raw)
    return out


# ────────────────────────────────
# KYC / COLLATERAL HELPERS
# ────────────────────────────────

def build_session_kyc_registry(force: bool = False) -> pd.DataFrame:
    n = 200
    rng = np.random.default_rng(42 if not force else None)
    df = pd.DataFrame(
        {
            "profile_id": [f"KYC{i:04d}" for i in range(1, n + 1)],
            "kyc_status": rng.choice(
                ["Cleared", "Enhanced Due Diligence", "Pending Docs"],
                p=[0.65, 0.1, 0.25],
                size=n,
            ),
            "aml_risk": rng.choice(["Low", "Medium", "High", "Critical"], p=[0.6, 0.25, 0.1, 0.05], size=n),
            "pep_status": rng.choice(["No match", "Match"], p=[0.95, 0.05], size=n),
            "watchlist_hits": rng.integers(0, 3, size=n),
            "next_refresh_due": pd.Timestamp.today()
            + pd.to_timedelta(rng.integers(0, 180, size=n), unit="D"),
        }
    )
    st.session_state["kyc_registry_ready"] = df.copy()
    st.session_state["kyc_registry_generated_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return df


def build_collateral_report(
    df: pd.DataFrame,
    *,
    confidence_threshold: float = 0.88,
    value_ratio: float = 0.8,
) -> Tuple[pd.DataFrame, List[str]]:
    if df is None or df.empty:
        return pd.DataFrame(), []
    records = df.to_dict(orient="records")
    progress = st.progress(0.0) if len(records) > 1 else None
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    session = requests.Session()
    for idx, record in enumerate(records, start=1):
        asset_type = str(record.get("collateral_type") or "Collateral Asset")
        declared_value = _to_float(record.get("collateral_value"), 0.0)
        loan_amount = _to_float(record.get("loan_amount"), 0.0)
        metadata = {
            "application_id": record.get("application_id"),
            "declared_value": declared_value,
            "loan_amount": loan_amount,
            "currency_code": record.get("currency_code")
            or st.session_state.get("currency_code"),
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

        value_threshold = (
            loan_amount * value_ratio
            if loan_amount
            else declared_value * value_ratio
        )
        meets_confidence = confidence >= confidence_threshold
        meets_value = value_threshold == 0 or estimated_value >= value_threshold
        reasons: List[str] = []
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


# ────────────────────────────────
# DASHBOARDS
# ────────────────────────────────

def render_credit_dashboard(df: pd.DataFrame, currency_symbol: str = "") -> None:
    if df is None or df.empty:
        st.info("No data to chart yet.")
        return
    if "decision" in df.columns:
        fig1 = px.histogram(df, x="decision", title="Decision Distribution")
        st.plotly_chart(fig1, use_container_width=True)
    if "loan_amount" in df.columns:
        fig2 = px.histogram(
            df,
            x="loan_amount",
            nbins=30,
            title=f"Loan Amounts {currency_symbol}".strip(),
        )
        st.plotly_chart(fig2, use_container_width=True)


# ────────────────────────────────
# PAGE RENDERERS (LANDING / AGENTS / LOGIN)
# ────────────────────────────────

def render_landing() -> None:
    col1, col2 = st.columns([1.1, 1.9], gap="large")
    with col1:
        st.markdown("<div class='left-box'>", unsafe_allow_html=True)
        logo_path = load_image("people_logo")
        if logo_path:
            st.image(logo_path, width=160)
        else:
            logo_upload = st.file_uploader(
                "Upload People Logo",
                type=["jpg", "png", "webp"],
                key="upload_logo",
            )
            if logo_upload:
                save_uploaded_image(logo_upload, "people_logo")
                st.success("✅ Logo uploaded successfully! Refreshing...")
                st.rerun()
        st.markdown(
            """
            <h1>✊ Let’s Build an AI by the People, for the People</h1>
            <h3>⚙️ Ready-to-Use AI Agent Sandbox — From Sandbox to Production</h3>
            <p>
            A world-class open innovation space where anyone can build, test, and deploy AI agents using open-source code, explainable models, and modular templates.<br><br>
            For developers, startups, and enterprises — experiment, customize, and scale AI without barriers.<br><br>
            <b>Privacy & Data Sovereignty:</b> Each agent runs under strict privacy controls and complies with GDPR & Vietnam Data Law 2025. Only anonymized or synthetic data is used — your data never leaves your environment.<br><br>
            <b>From Sandbox to Production:</b> Start with ready-to-use agent templates, adapt, test, and deploy — on your infra or GPU-as-a-Service.<br><br>
            You dream it — now you can build it.
            </p>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🚀 Start Building Now", key="btn_start_build_now"):
            set_stage("agents", update_query=False)
            clear_query_params()
            st.rerun()
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
            image_html = render_image_tag(agent, industry, emoji)
            rows.append(
                {
                    "🖼️": image_html,
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


def render_agents() -> None:
    top = st.columns([1, 4, 1])
    with top[0]:
        if st.button("⬅️ Back to Home", use_container_width=True):
            st.session_state.stage = "landing"
            clear_query_params()
            st.rerun()
    with top[1]:
        st.title("🤖 Available AI Agents")
    agent_rows = [
        {
            "Agent": "💳 Credit Appraisal Agent",
            "Description": "Explainable AI for retail loan decisioning",
            "Status": "✅ Available",
            "Action": "<a class='macbtn' href='?stage=login&agent=credit'>🚀 Launch</a>",
        },
        {
            "Agent": "🏛️ Asset Appraisal Agent",
            "Description": "Market-driven collateral valuation",
            "Status": "🕓 Coming Soon",
            "Action": "—",
        },
    ]
    st.write(
        pd.DataFrame(agent_rows).to_html(escape=False, index=False),
        unsafe_allow_html=True,
    )
    qp_stage = st.query_params.get("stage") if hasattr(st, "query_params") else None
    if isinstance(qp_stage, list):
        qp_stage = qp_stage[0] if qp_stage else None
    if qp_stage == "login":
        st.session_state.stage = "login"
        clear_query_params()
        st.rerun()
    st.markdown("<footer>Made with ❤️ by Dzoan Nguyen — Open AI Sandbox Initiative</footer>", unsafe_allow_html=True)


def render_login() -> None:
    top = st.columns([1, 4, 1])
    with top[0]:
        if st.button("⬅️ Back to Agents", use_container_width=True):
            st.session_state.stage = "agents"
            st.rerun()
    with top[1]:
        st.title("🔐 Login to AI Credit Appraisal Platform")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        username = st.text_input("Username", placeholder="e.g. dzoan")
    with col2:
        email = st.text_input("Email", placeholder="e.g. dzoan@demo.local")
    with col3:
        password = st.text_input("Password", type="password", placeholder="Enter any password")
    if st.button("Login", use_container_width=True):
        if username.strip() and email.strip():
            st.session_state.user_info.update(
                {
                    "name": username.strip(),
                    "email": email.strip(),
                    "flagged": False,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            st.session_state.logged_in = True
            st.session_state.workflow_stage = "data"
            st.session_state.stage = "credit_agent"
            st.session_state["login_flash"] = username.strip()
            st.rerun()
        else:
            st.error("Please enter both username and email to continue.")
    st.markdown("<footer>Made with ❤️ by Dzoan Nguyen — Open AI Sandbox Initiative</footer>", unsafe_allow_html=True)


# ────────────────────────────────
# PIPELINE PAGE RENDERERS
# ────────────────────────────────

def render_pipeline_hero(active_stage: str) -> None:
    steps_html = "".join(
        f"""
        <div class='pipeline-step{' pipeline-step--active' if key == active_stage else ''}'>
            <div class='pipeline-step__index{' pipeline-step__index--active' if key == active_stage else ''}'>{idx}</div>
            <div class='pipeline-step__body'>
                <div class='pipeline-step__title'>{title}</div>
                <p>{desc}</p>
            </div>
        </div>
        """
        for idx, (key, title, desc) in enumerate(PIPELINE_STAGES, start=1)
    )
    st.markdown(
        f"""
        <div class='pipeline-hero'>
            <div class='pipeline-hero__header'>
                <div class='pipeline-hero__eyebrow'>Agent Workflow</div>
                <h1>💳 AI Credit Appraisal Platform</h1>
                <p>Generate, sanitize, and appraise credit with AI agent power and human decisions.</p>
            </div>
            <div class='pipeline-steps'>
                {steps_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_data() -> None:
    render_pipeline_hero("data")
    st.title("🏗️ Data Stage")
    st.caption("Generate or upload data, anonymize, and prepare for KYC/Asset/Credit stages.")
    col_currency, col_hint = st.columns([1, 2])
    with col_currency:
        options = list(CURRENCY_OPTIONS.keys())
        selection = st.selectbox(
            "Currency",
            options,
            index=options.index(st.session_state["currency_code_label"]),
        )
        if selection != st.session_state["currency_code_label"]:
            st.session_state["currency_code_label"] = selection
            set_currency_defaults()
    with col_hint:
        st.info(
            f"Amounts will be generated in **{st.session_state['currency_code']}**.",
            icon="💰",
        )
    rows = st.slider("Number of rows to generate", 50, 2000, 200, step=50)
    non_bank_ratio = st.slider("Share of non-bank customers", 0.0, 1.0, 0.30, 0.05)
    col_raw, col_anon = st.columns(2)
    with col_raw:
        if st.button("🔴 Generate RAW Synthetic Data (with PII)", use_container_width=True):
            raw_df = generate_raw_synthetic(rows, non_bank_ratio)
            st.session_state.synthetic_raw_df = raw_df
            raw_path = save_to_runs(raw_df, "synthetic_raw")
            st.success(f"Generated RAW (PII) dataset with {rows} rows. Saved to {raw_path}")
            st.dataframe(raw_df.head(10), use_container_width=True)
            st.download_button(
                "⬇️ Download RAW CSV",
                raw_df.to_csv(index=False).encode("utf-8"),
                os.path.basename(raw_path),
                "text/csv",
            )
    with col_anon:
        if st.button("🟢 Generate ANON Synthetic Data (ready for agent)", use_container_width=True):
            anon_df = generate_anon_synthetic(rows, non_bank_ratio)
            st.session_state.synthetic_df = anon_df
            anon_path = save_to_runs(anon_df, "synthetic_anon")
            st.success(f"Generated ANON dataset with {rows} rows. Saved to {anon_path}")
            st.dataframe(anon_df.head(10), use_container_width=True)
            st.download_button(
                "⬇️ Download ANON CSV",
                anon_df.to_csv(index=False).encode("utf-8"),
                os.path.basename(anon_path),
                "text/csv",
            )
    st.markdown("---")
    st.subheader("🧹 Upload & Anonymize Customer Data")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"], key="data_stage_uploader")
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            return
        st.write("📊 Original Data Preview:")
        st.dataframe(dedupe_columns(df.head(5)), use_container_width=True)
        sanitized, dropped_cols = drop_pii_columns(df)
        st.session_state.anonymized_df = sanitized
        dropped_text = ", ".join(dropped_cols) if dropped_cols else "none"
        st.success(f"Dropped possible PII columns: {dropped_text}")
        st.dataframe(sanitized.head(10), use_container_width=True)
    nav = st.columns([1, 1, 1])
    with nav[1]:
        if st.button("➡️ Continue to KYC"):
            st.session_state.workflow_stage = "kyc"
            st.rerun()


def page_kyc() -> None:
    render_pipeline_hero("kyc")
    st.title("🛂 KYC & Compliance Workbench")
    st.caption(
        "Capture applicant identity, perform sanctions checks, and feed compliance context downstream."
    )
    if st.button("🔁 Refresh Synthetic KYC Dossier"):
        build_session_kyc_registry(force=True)
        st.success("Synthetic KYC dossier refreshed.")
        st.rerun()
    kyc_df = st.session_state.get("kyc_registry_ready")
    if kyc_df is None:
        kyc_df = build_session_kyc_registry()
    generated_at = st.session_state.get("kyc_registry_generated_at")
    st.markdown(f"**Synthetic dossier ready · Last refresh {generated_at}**")
    st.dataframe(kyc_df.head(15), use_container_width=True)
    st.markdown("#### 📥 Anonymized KYC (credit-ready)")
    st.dataframe(kyc_df.head(15), use_container_width=True)
    nav = st.columns([1, 1, 1])
    with nav[0]:
        if st.button("⬅️ Back to Data Stage"):
            st.session_state.workflow_stage = "data"
            st.rerun()
    with nav[1]:
        if st.button("🏛️ Continue to Asset Stage"):
            st.session_state.workflow_stage = "asset"
            st.rerun()


def page_asset() -> None:
    render_pipeline_hero("asset")
    st.title("🏛️ Collateral Asset Platform")
    st.caption("Verify collateral assets in batch before running the credit appraisal agent.")
    options = [
        "Use synthetic (ANON)",
        "Use synthetic (RAW – auto-sanitize)",
        "Use anonymized dataset",
        "Upload manually",
    ]
    choice = st.selectbox("Collateral data source", options, key="asset_data_choice")
    dataset_preview: Optional[pd.DataFrame] = None
    if choice == "Use synthetic (ANON)":
        dataset_preview = st.session_state.get("synthetic_df")
    elif choice == "Use synthetic (RAW – auto-sanitize)":
        raw = st.session_state.get("synthetic_raw_df")
        if raw is not None:
            dataset_preview, _ = drop_pii_columns(raw)
    elif choice == "Use anonymized dataset":
        dataset_preview = st.session_state.get("anonymized_df")
    elif choice == "Upload manually":
        uploaded = st.file_uploader(
            "Upload CSV for collateral verification",
            type=["csv"],
            key="asset_manual_upload",
        )
        if uploaded is not None:
            try:
                dataset_preview = pd.read_csv(uploaded)
                st.success(f"Staged `{uploaded.name}` for collateral verification.")
            except Exception as exc:
                st.error(f"Could not read CSV: {exc}")
    if dataset_preview is not None and not dataset_preview.empty:
        with st.expander("Preview selected dataset", expanded=False):
            st.dataframe(ensure_application_ids(dataset_preview).head(10), use_container_width=True)
    else:
        st.info("Select or generate a dataset to begin collateral verification.", icon="ℹ️")
    col_conf, col_ratio = st.columns(2)
    with col_conf:
        confidence_threshold = st.slider(
            "Minimum confidence from asset agent", 0.50, 1.00, 0.88, 0.01
        )
    with col_ratio:
        value_ratio = st.slider(
            "Min estimated collateral vs. loan ratio", 0.10, 1.50, 0.80, 0.05
        )
    if st.button("🛡️ Generate collateral verification report", use_container_width=True):
        if dataset_preview is None or dataset_preview.empty:
            st.warning("No dataset available. Generate synthetic data or upload a CSV first.")
        else:
            required = {"application_id", "collateral_type", "collateral_value"}
            missing = [c for c in required if c not in dataset_preview.columns]
            if missing:
                st.error("Dataset is missing required columns: " + ", ".join(sorted(missing)))
            else:
                with st.spinner("Running asset appraisal agent across collateral records..."):
                    report_df, _ = build_collateral_report(
                        ensure_application_ids(dataset_preview),
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
    nav = st.columns([1, 1, 1])
    with nav[0]:
        if st.button("⬅️ Back to KYC"):
            st.session_state.workflow_stage = "kyc"
            st.rerun()
    with nav[1]:
        if st.button("➡️ Continue to Credit"):
            if st.session_state.get("asset_collateral_df") is None:
                st.warning("Run the asset appraisal first.")
            else:
                st.session_state.workflow_stage = "credit"
                st.rerun()


def page_credit() -> None:
    render_pipeline_hero("credit")
    st.title("🤖 Credit Appraisal")
    st.caption("Run the credit agent on your dataset and view dashboards.")
    data_options = [
        "Use synthetic (ANON)",
        "Use synthetic (RAW – auto-sanitize)",
        "Use anonymized dataset",
        "Use collateral verification output",
        "Upload manually",
    ]
    data_choice = st.selectbox("Select Data Source", data_options)
    if data_choice == "Upload manually":
        up = st.file_uploader("Upload your CSV", type=["csv"], key="manual_upload_run_file")
        if up is not None:
            st.session_state.manual_upload_name = up.name
            st.session_state.manual_upload_bytes = up.getvalue()
            st.success(f"File staged: {up.name}")
    rule_mode = st.radio(
        "Choose rule mode",
        ["Classic (bank-style metrics)", "NDI (Net Disposable Income) — simple"],
        index=0,
    )
    if st.button("🚀 Run Agent", use_container_width=True):
        df: Optional[pd.DataFrame] = None
        if data_choice == "Use synthetic (ANON)":
            df = st.session_state.get("synthetic_df")
        elif data_choice == "Use synthetic (RAW – auto-sanitize)":
            raw = st.session_state.get("synthetic_raw_df")
            if raw is not None:
                df, _ = drop_pii_columns(raw)
        elif data_choice == "Use anonymized dataset":
            df = st.session_state.get("anonymized_df")
        elif data_choice == "Use collateral verification output":
            df = st.session_state.get("asset_collateral_df")
        elif data_choice == "Upload manually":
            up_bytes = st.session_state.get("manual_upload_bytes")
            if up_bytes:
                try:
                    df = pd.read_csv(io.BytesIO(up_bytes))
                except Exception as exc:
                    st.error(f"Could not read uploaded CSV: {exc}")
        if df is None or df.empty:
            st.warning("No dataset available to run.")
        else:
            df = ensure_application_ids(df)
            rng = np.random.default_rng(7)
            df_out = df.copy()
            df_out["score"] = rng.uniform(0, 1, size=len(df_out))
            df_out["decision"] = np.where(df_out["score"] >= 0.5, "Approve", "Reject")
            st.session_state.last_merged_df = df_out
            st.success("✅ Run succeeded (simulated).")
            st.dataframe(df_out.head(20), use_container_width=True)
            render_credit_dashboard(df_out, st.session_state.get("currency_symbol", ""))
    if st.session_state.get("last_merged_df") is not None:
        st.markdown("---")
        st.subheader("📥 Download Latest Outputs")
        df_out = st.session_state["last_merged_df"]
        csv_bytes = df_out.to_csv(index=False).encode("utf-8")
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        st.download_button(
            "⬇️ Download CSV",
            csv_bytes,
            f"ai-appraisal-outputs-{ts}.csv",
            "text/csv",
        )
    nav = st.columns([1, 1, 1])
    with nav[0]:
        if st.button("⬅️ Back to Asset"):
            st.session_state.workflow_stage = "asset"
            st.rerun()
    with nav[1]:
        if st.button("➡️ Continue to Human Review"):
            st.session_state.workflow_stage = "review"
            st.rerun()


def page_review() -> None:
    render_pipeline_hero("review")
    st.title("🧑‍⚖️ Human Review")
    st.caption("Audit AI outputs, adjust verdicts, and capture agreement metrics.")
    uploaded_review = st.file_uploader(
        "Load AI outputs CSV for review (optional)",
        type=["csv"],
        key="review_csv_loader_stage",
    )
    if uploaded_review is not None:
        try:
            st.session_state.last_merged_df = pd.read_csv(uploaded_review)
            st.success("Loaded review dataset from uploaded CSV.")
        except Exception as exc:
            st.error(f"Could not read uploaded CSV: {exc}")
    if "last_merged_df" not in st.session_state:
        st.info("Run the agent (credit stage) or upload an AI outputs CSV to load results for review.")
        return
    dfm = st.session_state["last_merged_df"].copy()
    if "decision" not in dfm.columns:
        st.warning("No decision column found. Nothing to review.")
        return
    st.markdown("#### 1) Select rows to review and correct")
    editable = dfm[["application_id", "decision"]].copy()
    editable.rename(columns={"decision": "ai_decision"}, inplace=True)
    editable["human_decision"] = editable["ai_decision"]
    edited = st.data_editor(
        editable,
        num_rows="dynamic",
        use_container_width=True,
        key="review_editor",
    )
    if st.button("💾 Save corrections"):
        st.session_state.review_corrections = edited
        st.success("Corrections saved in session.")
    nav = st.columns([1, 1, 1])
    with nav[0]:
        if st.button("⬅️ Back to Credit"):
            st.session_state.workflow_stage = "credit"
            st.rerun()
    with nav[1]:
        if st.button("➡️ Continue to Training"):
            st.session_state.workflow_stage = "training"
            st.rerun()


def page_training() -> None:
    render_pipeline_hero("training")
    st.title("🔁 Training (Feedback → Retrain)")
    st.caption("Loop curated feedback into retraining jobs and promote production-ready models.")
    corr = st.session_state.get("review_corrections")
    if corr is not None:
        st.write("Recent corrections:")
        st.dataframe(corr, use_container_width=True)
    else:
        st.info("No corrections captured yet.")
    if st.button("📦 Export training dataset (CSV)"):
        base = st.session_state.get("last_merged_df")
        if base is None or base.empty:
            st.warning("No base run to export.")
        else:
            out = base.copy()
            if corr is not None and {
                "application_id",
                "human_decision",
            }.issubset(corr.columns):
                out = out.merge(
                    corr[["application_id", "human_decision"]],
                    on="application_id",
                    how="left",
                )
            buf = io.StringIO()
            out.to_csv(buf, index=False)
            st.download_button(
                "⬇️ Download Training CSV",
                buf.getvalue().encode("utf-8"),
                "training_dataset.csv",
                "text/csv",
            )
    nav = st.columns([1, 1, 1])
    with nav[0]:
        if st.button("⬅️ Back to Review"):
            st.session_state.workflow_stage = "review"
            st.rerun()
    with nav[1]:
        if st.button("🔄 Continue to Loop Back"):
            st.session_state.workflow_stage = "loopback"
            st.rerun()


def page_loopback() -> None:
    render_pipeline_hero("loopback")
    st.title("🔄 Loop Back to Step 3 → Use New Trained Model")
    st.caption(
        "Review production metadata and relaunch the credit appraisal stage with the latest promoted model."
    )
    st.markdown("### 📦 Production model status")
    try:
        resp = requests.get(f"{API_URL}/v1/training/production_meta", timeout=8)
        if resp.ok:
            meta = resp.json()
            if meta:
                st.json(meta)
            else:
                st.info("No production metadata returned yet.")
        else:
            st.warning(f"Could not fetch production metadata (status {resp.status_code}).")
    except Exception as exc:
        st.warning(f"Production metadata unavailable: {exc}")
    st.markdown("---")
    st.markdown(
        """
        **Next steps**

        1. Validate the promoted model's metadata and deployment status above.
        2. Return to the credit appraisal stage to generate fresh decisions with the updated model.
        3. Repeat the workflow to continuously improve decision quality.
        """
    )
    nav = st.columns([1, 1, 1])
    with nav[0]:
        if st.button("⬅️ Back to Training"):
            st.session_state.workflow_stage = "training"
            st.rerun()
    with nav[1]:
        if st.button("➡️ Continue to Credit Stage"):
            st.session_state.workflow_stage = "credit"
            st.rerun()


# ────────────────────────────────
# WORKFLOW SHELL
# ────────────────────────────────

def render_workflow() -> None:
    top = st.columns([1, 3, 1])
    with top[0]:
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.stage = "landing"
            clear_query_params()
            st.session_state.logged_in = False
            st.rerun()
    with top[1]:
        user = st.session_state.get("user_info", {}).get("name", "")
        if flash := st.session_state.pop("login_flash", None):
            st.success(f"✅ Logged in as {flash}")
        st.caption(f"Logged in as **{user}**")
    with top[2]:
        if st.button("🚪 Logout", use_container_width=True):
            logout_user()
    stage = st.session_state.get("workflow_stage", "data")
    if stage == "data":
        page_data()
    elif stage == "kyc":
        page_kyc()
    elif stage == "asset":
        page_asset()
    elif stage == "credit":
        page_credit()
    elif stage == "review":
        page_review()
    elif stage == "training":
        page_training()
    elif stage == "loopback":
        page_loopback()
    else:
        st.session_state.workflow_stage = "data"
        page_data()
    st.markdown(
        "<footer>Made with ❤️ by Dzoan Nguyen — Open AI Sandbox Initiative</footer>",
        unsafe_allow_html=True,
    )


# ────────────────────────────────
# ROUTER
# ────────────────────────────────

def main() -> None:
    stage = st.session_state.get("stage", "landing")
    if stage == "landing":
        render_landing()
        return
    if stage == "agents":
        render_agents()
        return
    if stage == "login":
        render_login()
        return
    if stage == "credit_agent":
        if not st.session_state.get("logged_in"):
            set_stage("login")
            st.rerun()
            return
        render_workflow()
        return
    set_stage("landing")
    render_landing()


if __name__ == "__main__":
    main()
