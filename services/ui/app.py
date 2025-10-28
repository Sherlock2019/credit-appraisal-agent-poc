# services/ui/app.py — Fixed, self-contained version
# ─────────────────────────────────────────────
# 🌐 OpenSource AI Agent Library + Credit Appraisal PoC by Dzoan (fixed skeleton)
# ─────────────────────────────────────────────
from __future__ import annotations
import os
import io
import json
import datetime
from typing import Optional, Dict, List, Any

import numpy as np
import pandas as pd
import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go

# ────────────────────────────────
# GLOBAL CONFIG
# ────────────────────────────────
BASE_DIR = os.path.expanduser("~/credit-appraisal-agent-poc/services/ui")
LANDING_IMG_DIR = os.path.join(BASE_DIR, "landing_images")
RUNS_DIR = os.path.join(BASE_DIR, ".runs")
TMP_FEEDBACK_DIR = os.path.join(BASE_DIR, ".tmp_feedback")

for d in (LANDING_IMG_DIR, RUNS_DIR, TMP_FEEDBACK_DIR):
    os.makedirs(d, exist_ok=True)

API_URL = os.getenv("API_URL", "http://localhost:8090")

# ────────────────────────────────
# SESSION STATE INIT
# ────────────────────────────────
if "stage" not in st.session_state:
    st.session_state.stage = "landing"
if "user_info" not in st.session_state:
    st.session_state.user_info = {"name": "", "email": "", "flagged": False}
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "flagged" not in st.session_state.user_info:
    st.session_state.user_info["flagged"] = False
if "timestamp" not in st.session_state.user_info:
    st.session_state.user_info["timestamp"] = datetime.datetime.utcnow().isoformat()

# ────────────────────────────────
# PAGE CONFIG
# ────────────────────────────────
st.set_page_config(
    page_title="AI Agent Sandbox — By the People, For the People",
    layout="wide",
)

# ────────────────────────────────
# UTILITIES & STUBS (safe fallbacks)
# ────────────────────────────────
def _clear_qp():
    """Clear query params (modern Streamlit API)."""
    try:
        st.query_params.clear()
    except Exception:
        pass


def load_image(base: str) -> Optional[str]:
    for ext in [".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"]:
        p = os.path.join(LANDING_IMG_DIR, f"{base}{ext}")
        if os.path.exists(p):
            return p
    return None


def save_uploaded_image(uploaded_file, base: str):
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
            f'<img src="file://{img_path}" '
            f'style="width:48px;height:48px;border-radius:10px;object-fit:cover;">'
        )
    return f'<div style="font-size:32px;">{emoji_fallback}</div>'


# ────────────────────────────────
# DATA
# ────────────────────────────────
AGENTS = [
    ("🏦 Banking & Finance", "💰 Retail Banking", "💳 Credit Appraisal Agent",
     "Explainable AI for loan decisioning", "Available", "💳"),
    ("🏦 Banking & Finance", "💰 Retail Banking", "🏦 Asset Appraisal Agent",
     "Market-driven collateral valuation", "Coming Soon", "🏦"),
    ("🏦 Banking & Finance", "🩺 Insurance", "🩺 Claims Triage Agent",
     "Automated claims prioritization", "Coming Soon", "🩺"),
    ("⚡ Energy & Sustainability", "🔋 EV & Charging", "⚡ EV Charger Optimizer",
     "Optimize charger deployment via AI", "Coming Soon", "⚡"),
    ("⚡ Energy & Sustainability", "☀️ Solar", "☀️ Solar Yield Estimator",
     "Estimate solar ROI and efficiency", "Coming Soon", "☀️"),
    ("🚗 Automobile & Transport", "🚙 Automobile", "🚗 Predictive Maintenance",
     "Prevent downtime via sensor analytics", "Coming Soon", "🚗"),
    ("🚗 Automobile & Transport", "🔋 EV", "🔋 EV Battery Health Agent",
     "Monitor EV battery health cycles", "Coming Soon", "🔋"),
    ("🚗 Automobile & Transport", "🚚 Ride-hailing / Logistics", "🛻 Fleet Route Optimizer",
     "Dynamic route optimization for fleets", "Coming Soon", "🛻"),
    ("💻 Information Technology", "🧰 Support & Security", "🧩 IT Ticket Triage",
     "Auto-prioritize support tickets", "Coming Soon", "🧩"),
    ("💻 Information Technology", "🛡️ Security", "🔐 SecOps Log Triage",
     "Detect anomalies & summarize alerts", "Coming Soon", "🔐"),
    ("⚖️ Legal & Government", "⚖️ Law Firms", "⚖️ Contract Analyzer",
     "Extract clauses and compliance risks", "Coming Soon", "⚖️"),
    ("⚖️ Legal & Government", "🏛️ Public Services", "🏛️ Citizen Service Agent",
     "Smart assistant for citizen services", "Coming Soon", "🏛️"),
    ("🛍️ Retail / SMB / Creative", "🏬 Retail & eCommerce", "📈 Sales Forecast Agent",
     "Predict demand & inventory trends", "Coming Soon", "📈"),
    ("🎬 Retail / SMB / Creative", "🎨 Media & Film", "🎬 Budget Cost Assistant",
     "Estimate, optimize, and track film & production costs using AI", "Coming Soon", "🎬"),
]

# ────────────────────────────────
# STYLES
# ────────────────────────────────
st.markdown(
    """
    <style>
    html, body, .block-container { background-color:#0f172a !important; color:#e2e8f0 !important; }
    footer { text-align:center; padding:2rem; color:#aab3c2; font-size:1.2rem; font-weight:600; margin-top:2rem; }
    .left-box {
        background: radial-gradient(circle at top left, #0f172a, #1e293b);
        border-radius:20px; padding:3rem 2rem; color:#f1f5f9; box-shadow:6px 0 24px rgba(0,0,0,0.4);
    }
    .right-box {
        background:linear-gradient(180deg,#1e293b,#0f172a);
        border-radius:20px; padding:2rem; box-shadow:-6px 0 24px rgba(0,0,0,0.35);
    }
    .stButton > button {
        border:none !important; cursor:pointer;
        padding:14px 28px !important; font-size:18px !important; font-weight:700 !important;
        border-radius:14px !important; color:#fff !important;
        background:linear-gradient(180deg,#4ea3ff 0%,#2f86ff 60%,#0f6fff 100%) !important;
        box-shadow:0 8px 24px rgba(15,111,255,0.35);
    }
    a.macbtn {
        display:inline-block; text-decoration:none !important; color:#fff !important;
        padding:10px 22px; font-weight:700; border-radius:12px;
        background:linear-gradient(180deg,#4ea3ff 0%,#2f86ff 60%,#0f6fff 100%);
    }
    /* Larger workflow tabs */
    [data-testid="stTabs"] [data-baseweb="tab"] {
        font-size: 28px !important;
        font-weight: 800 !important;
        padding: 20px 40px !important;
        border-radius: 12px !important;
        background-color: #1e293b !important;
        color: #f8fafc !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(90deg, #2563eb, #1d4ed8) !important;
        color: white !important;
        border-bottom: 6px solid #60a5fa !important;
        box-shadow: 0 4px 14px rgba(37,99,235,0.5);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────
# QUERY PARAM ROUTING (modern API)
# ────────────────────────────────
try:
    qp = st.query_params
except Exception:
    qp = {}

if "stage" in qp:
    target = qp["stage"]
    if target in {"landing", "agents", "login", "credit_agent"} and st.session_state.stage != target:
        st.session_state.stage = target
        _clear_qp()
        st.rerun()

if "launch" in qp or ("agent" in qp and qp.get("agent") == ["credit"]):
    st.session_state.stage = "login"
    _clear_qp()
    st.rerun()

# ────────────────────────────────
# STAGE: LANDING
# ────────────────────────────────
if st.session_state.stage == "landing":
    c1, c2 = st.columns([1.1, 1.9], gap="large")
    with c1:
        st.markdown("<div class='left-box'>", unsafe_allow_html=True)
        logo_path = load_image("people_logo")
        if logo_path:
            st.image(logo_path, width=160)
        else:
            up = st.file_uploader("Upload People Logo", type=["jpg", "png", "webp"], key="upload_logo")
            if up:
                save_uploaded_image(up, "people_logo")
                st.success("✅ Logo uploaded, refresh to view.")
        st.markdown(
            """
            <h1>✊ Let’s Build an AI by the People, for the People</h1>
            <h3>⚙️ Ready-to-Use AI Agent Sandbox — From Sandbox to Production</h3>
            <p>Build, test, and deploy AI agents using open-source explainable models.<br><br>
            <b>Privacy:</b> Synthetic & anonymized data only.<br>
            <b>Deployment:</b> GPU-as-a-Service Cloud, zero CAPEX.</p>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🚀 Start Building Now", key="btn_start_build_now"):
            st.session_state.stage = "agents"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='right-box'>", unsafe_allow_html=True)
        st.markdown("<h2>📊 Global AI Agent Library</h2>", unsafe_allow_html=True)
        rows = []
        for sector, industry, agent, desc, status, emoji in AGENTS:
            rows.append({
                "🖼️": render_image_tag(agent, industry, emoji),
                "🏭 Sector": sector,
                "🧩 Industry": industry,
                "🤖 Agent": agent,
                "🧠 Description": desc,
                "📶 Status": f'<span style="color:{"#22c55e" if status=="Available" else "#f59e0b"};">{status}</span>'
            })
        st.write(pd.DataFrame(rows).to_html(escape=False, index=False), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<footer>Made with ❤️ by Dzoan Nguyen — Open AI Sandbox Initiative</footer>", unsafe_allow_html=True)
    st.stop()

# ────────────────────────────────
# STAGE: AGENTS
# ────────────────────────────────
if st.session_state.stage == "agents":
    top = st.columns([1, 4, 1])
    with top[0]:
        if st.button("⬅️ Back to Home", key="btn_back_home_from_agents"):
            st.session_state.stage = "landing"
            st.rerun()
    with top[1]:
        st.title("🤖 Available AI Agents")

    df = pd.DataFrame([
        {"Agent": "💳 Credit Appraisal Agent",
         "Description": "Explainable AI for retail loan decisioning",
         "Status": "✅ Available",
         "Action": '<a class="macbtn" href="?agent=credit&stage=login">🚀 Launch</a>'},
        {"Agent": "🏦 Asset Appraisal Agent",
         "Description": "Market-driven collateral valuation",
         "Status": "🕓 Coming Soon", "Action": "—"},
    ])
    st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)
    st.markdown("<footer>Made with ❤️ by Dzoan Nguyen — Open AI Sandbox Initiative</footer>", unsafe_allow_html=True)
    st.stop()

# ────────────────────────────────
# STAGE: LOGIN
# ────────────────────────────────
if st.session_state.stage == "login":
    top = st.columns([1, 4, 1])
    with top[0]:
        if st.button("⬅️ Back to Agents", key="btn_back_agents_from_login"):
            st.session_state.stage = "agents"
            st.rerun()
    with top[1]:
        st.title("🔐 Login to AI Credit Appraisal Platform")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        user = st.text_input("Username", placeholder="e.g. dzoan")
    with c2:
        email = st.text_input("Email", placeholder="e.g. dzoan@demo.local")
    with c3:
        pwd = st.text_input("Password", type="password", placeholder="Enter any password")
    if st.button("Login", key="btn_login_submit", use_container_width=True):
        if user.strip() and email.strip():
            st.session_state.user_info = {
                "name": user.strip(),
                "email": email.strip(),
                "flagged": False,
                "timestamp": datetime.datetime.utcnow().isoformat()
            }
            st.session_state.logged_in = True
            st.session_state.stage = "credit_agent"
            st.rerun()
        else:
            st.error("⚠️ Please fill all fields before continuing.")
    st.markdown("<footer>Made with ❤️ by Dzoan Nguyen — Open AI Sandbox Initiative</footer>", unsafe_allow_html=True)
    st.stop()

# ────────────────────────────────
# STAGE: CREDIT WORKFLOW
# ────────────────────────────────
if st.session_state.stage == "credit_agent":
    top = st.columns([1, 4, 1])
    with top[0]:
        if st.button("⬅️ Back to Agents", key="btn_back_agents_from_pipeline"):
            st.session_state.stage = "agents"
            st.rerun()
    with top[1]:
        st.title("💳 AI Credit Appraisal Platform")
        st.caption("Generate, sanitize, and appraise credit with AI agent power and human insight.")




# ────────────────────────────────
# GLOBAL UTILS
# ────────────────────────────────


BANNED_NAMES = {"race", "gender", "religion", "ethnicity", "ssn", "national_id"}
PII_COLS = {"customer_name", "name", "email", "phone", "address", "ssn", "national_id", "dob"}

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d\-\s]{6,}\d")

def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.duplicated(keep="last")]

def scrub_text_pii(s):
    if not isinstance(s, str):
        return s
    s = EMAIL_RE.sub("", s)
    s = PHONE_RE.sub("", s)
    return s.strip()

def drop_pii_columns(df: pd.DataFrame):
    original_cols = list(df.columns)
    keep_cols = [c for c in original_cols if all(k not in c.lower() for k in PII_COLS)]
    dropped = [c for c in original_cols if c not in keep_cols]
    out = df[keep_cols].copy()
    for c in out.select_dtypes(include="object"):
        out[c] = out[c].apply(scrub_text_pii)
    return dedupe_columns(out), dropped

def strip_policy_banned(df: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for c in df.columns:
        cl = c.lower()
        if cl in BANNED_NAMES:
            continue
        keep.append(c)
    return df[keep]

def append_user_info(df: pd.DataFrame) -> pd.DataFrame:
    meta = st.session_state["user_info"]
    out = df.copy()
    out["session_user_name"] = meta["name"]
    out["session_user_email"] = meta["email"]
    out["session_flagged"] = meta["flagged"]
    out["created_at"] = meta["timestamp"]
    return dedupe_columns(out)

def save_to_runs(df: pd.DataFrame, prefix: str) -> str:
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    flag_suffix = "_FLAGGED" if st.session_state["user_info"]["flagged"] else ""
    fname = f"{prefix}_{ts}{flag_suffix}.csv"
    fpath = os.path.join(RUNS_DIR, fname)
    dedupe_columns(df).to_csv(fpath, index=False)
    return fpath

def try_json(x):
    if isinstance(x, (dict, list)):
        return x
    if not isinstance(x, str):
        return None
    try:
        return json.loads(s)
    except Exception:
        return None

def _safe_json(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str) and x.strip():
        try:
            return json.loads(x)
        except Exception:
            return {}
    return {}

def fmt_currency_label(base: str) -> str:
    sym = st.session_state.get("currency_symbol", "")
    return f"{base} ({sym})" if sym else base

# ─────────────────────────────────────────────
# CURRENCY CATALOG

CURRENCY_OPTIONS = {
    # code: (label, symbol, fx to apply on USD-like base generated numbers)
    "USD": ("USD $", "$", 1.0),
    "EUR": ("EUR €", "€", 0.93),
    "GBP": ("GBP £", "£", 0.80),
    "JPY": ("JPY ¥", "¥", 150.0),
    "VND": ("VND ₫", "₫", 24000.0),
}

def set_currency_defaults():
    if "currency_code" not in st.session_state:
        st.session_state["currency_code"] = "USD"
    label, symbol, fx = CURRENCY_OPTIONS[st.session_state["currency_code"]]
    st.session_state["currency_label"] = label
    st.session_state["currency_symbol"] = symbol
    st.session_state["currency_fx"] = fx

set_currency_defaults()

# ─────────────────────────────────────────────
# DASHBOARD HELPERS (Plotly, dark theme)

def _kpi_card(label: str, value: str, sublabel: str | None = None):
    st.markdown(
        f"""
        <div style="background:#0e1117;border:1px solid #2a2f3e;border-radius:12px;padding:14px 16px;margin-bottom:10px;">
          <div style="font-size:12px;color:#9aa4b2;text-transform:uppercase;letter-spacing:.06em;">{label}</div>
          <div style="font-size:28px;font-weight:700;color:#e6edf3;line-height:1.1;margin-top:2px;">{value}</div>
          {f'<div style="font-size:12px;color:#9aa4b2;margin-top:6px;">{sublabel}</div>' if sublabel else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_credit_dashboard(df: pd.DataFrame, currency_symbol: str = ""):
    """
    Renders the whole dashboard (TOP-10s → Opportunities → KPIs & pies/bars → Mix table).
    Keeps decision filter in the table only.
    """
    if df is None or df.empty:
        st.info("No data to chart yet.")
        return
    # Basic charts: decision distribution and loan amount histogram
    if "decision" in df.columns:
        fig1 = px.histogram(df, x="decision", title="Decision Distribution")
        st.plotly_chart(fig1, use_container_width=True)
    if "loan_amount" in df.columns:
        fig2 = px.histogram(df, x="loan_amount", nbins=30, title=f"Loan Amounts {currency_symbol}".strip())
        st.plotly_chart(fig2, use_container_width=True)

def go_to_public_home(clear_user: bool = False) -> None:
    if clear_user:
        st.session_state.user_info = {"flagged": False}
        st.session_state.logged_in = False
    st.session_state.workflow_stage = "data"

def logout_user():
    st.session_state.clear()

def build_session_kyc_registry(force: bool = False) -> pd.DataFrame:
    # Synthetic KYC profiles — deterministic but simple
    n = 200
    rng = np.random.default_rng(42 if not force else None)
    df = pd.DataFrame({
        "profile_id": [f"KYC{i:04d}" for i in range(1, n+1)],
        "kyc_status": rng.choice(["Cleared", "Enhanced Due Diligence", "Pending Docs"], p=[0.65, 0.1, 0.25], size=n),
        "aml_risk": rng.choice(["Low", "Medium", "High", "Critical"], p=[0.6, 0.25, 0.1, 0.05], size=n),
        "pep_status": rng.choice(["No match", "Match"], p=[0.95, 0.05], size=n),
        "watchlist_hits": rng.integers(0, 3, size=n),
        "next_refresh_due": pd.Timestamp.today() + pd.to_timedelta(rng.integers(0, 180, size=n), unit="D"),
    })
    ready = df.copy()
    st.session_state["kyc_registry_ready"] = ready
    st.session_state["kyc_registry_generated_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return df

    cols = df.columns

    # ─────────────── TOP 10s FIRST ───────────────
    st.markdown("## 🔝 Top 10 Snapshot")

    # Top 10 loans approved
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

    # Top 10 collateral types by average value
    if {"collateral_type", "collateral_value"} <= set(cols):
        cprof = df.groupby("collateral_type", dropna=False).agg(
            avg_value=("collateral_value", "mean"),
            cnt=("collateral_type", "count")
        ).reset_index()
        if not cprof.empty:
            cprof = cprof.sort_values("avg_value", ascending=False).head(10)
            fig = px.bar(
                cprof,
                x="avg_value",
                y="collateral_type",
                orientation="h",
                title="Top 10 Collateral Types (Avg Value)",
                labels={"avg_value": f"Avg Value {currency_symbol}", "collateral_type": "Collateral Type"},
                hover_data=["cnt"]
            )
            fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    # Top 10 reasons for denial (from rule_reasons False flags)
    if "rule_reasons" in cols and "decision" in cols:
        denied = df[df["decision"].astype(str).str.lower() == "denied"].copy()
        reasons_count = {}
        for _, r in denied.iterrows():
            rr = _safe_json(r.get("rule_reasons"))
            if isinstance(rr, dict):
                for k, v in rr.items():
                    if v is False:
                        reasons_count[k] = reasons_count.get(k, 0) + 1
        if reasons_count:
            items = pd.DataFrame(sorted(reasons_count.items(), key=lambda x: x[1], reverse=True),
                                 columns=["reason", "count"]).head(10)
            fig = px.bar(
                items, x="count", y="reason", orientation="h",
                title="Top 10 Reasons for Denial",
                labels={"count": "Count", "reason": "Rule"},
            )
            fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No denial reasons detected.")

    # Top 10 loan officer performance (approval rate) if officer column present
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
                perf, x="approved_rate_pct", y=officer_col, orientation="h",
                title="Top 10 Loan Officer Approval Rate (this batch)",
                labels={"approved_rate_pct": "Approval Rate (%)", officer_col: "Officer"},
                hover_data=["n"]
            )
            fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ─────────────── OPPORTUNITIES ───────────────
    st.markdown("## 💡 Opportunities")

    # Short-term loan opportunities (simple heuristic)
    opp_rows = []
    if {"income", "loan_amount"}.issubset(cols):
        term_col = "loan_term_months" if "loan_term_months" in cols else ("loan_duration_months" if "loan_duration_months" in cols else None)
        if term_col:
            for _, r in df.iterrows():
                inc = float(r.get("income", 0) or 0)
                amt = float(r.get("loan_amount", 0) or 0)
                term = int(r.get(term_col, 0) or 0)
                dti = float(r.get("DTI", 0) or 0)
                if (term >= 36) and (amt <= inc * 0.8) and (dti <= 0.45):
                    opp_rows.append({
                        "application_id": r.get("application_id"),
                        "suggested_term": 24,
                        "loan_amount": amt,
                        "income": inc,
                        "DTI": dti,
                        "note": "Candidate for short-term plan (<=24m) based on affordability."
                    })
    if opp_rows:
        st.markdown("#### 📎 Short-Term Loan Candidates")
        st.dataframe(pd.DataFrame(opp_rows).head(25), use_container_width=True, height=320)
    else:
        st.info("No short-term loan candidates identified in this batch.")

    st.markdown("#### 🔁 Buyback / Consolidation Beneficiaries")
    candidates = []
    need = {"decision", "existing_debt", "loan_amount", "DTI"}
    if need <= set(cols):
        for _, r in df.iterrows():
            dec = str(r.get("decision", "")).lower()
            debt = float(r.get("existing_debt", 0) or 0)
            loan = float(r.get("loan_amount", 0) or 0)
            dti = float(r.get("DTI", 0) or 0)
            proposal = _safe_json(r.get("proposed_consolidation_loan", {}))
            has_bb = bool(proposal)

            if dec == "denied" or dti > 0.45 or debt > loan:
                benefit_score = round((debt / (loan + 1e-6)) * 0.4 + dti * 0.6, 2)
                candidates.append({
                    "application_id": r.get("application_id"),
                    "customer_type": r.get("customer_type"),
                    "existing_debt": debt,
                    "loan_amount": loan,
                    "DTI": dti,
                    "collateral_type": r.get("collateral_type"),
                    "buyback_proposed": has_bb,
                    "buyback_amount": proposal.get("buyback_amount") if has_bb else None,
                    "benefit_score": benefit_score,
                    "note": proposal.get("note") if has_bb else None
                })
    if candidates:
        cand_df = pd.DataFrame(candidates).sort_values("benefit_score", ascending=False)
        st.dataframe(cand_df.head(25), use_container_width=True, height=380)
    else:
        st.info("No additional buyback beneficiaries identified.")

    st.markdown("---")

    # ─────────────── PORTFOLIO KPIs ───────────────
    st.markdown("## 📈 Portfolio Snapshot")
    c1, c2, c3, c4 = st.columns(4)

    # Approval rate
    if "decision" in cols:
        total = len(df)
        approved = int((df["decision"].astype(str).str.lower() == "approved").sum())
        rate = (approved / total * 100) if total else 0.0
        with c1: _kpi_card("Approval Rate", f"{rate:.1f}%", f"{approved} of {total}")

    # Avg approved loan amount
    if {"decision", "loan_amount"} <= set(cols):
        ap = df[df["decision"].astype(str).str.lower() == "approved"]["loan_amount"]
        avg_amt = ap.mean() if len(ap) else 0.0
        with c2: _kpi_card("Avg Approved Amount", f"{currency_symbol}{avg_amt:,.0f}")

    # Decision time (if present)
    if {"created_at", "decision_at"} <= set(cols):
        try:
            t = (pd.to_datetime(df["decision_at"]) - pd.to_datetime(df["created_at"])).dt.total_seconds() / 60.0
            avg_min = float(t.mean())
            with c3: _kpi_card("Avg Decision Time", f"{avg_min:.1f} min")
        except Exception:
            with c3: _kpi_card("Avg Decision Time", "—")

    # Non-bank share
    if "customer_type" in cols:
        nb = int((df["customer_type"].astype(str).str.lower() == "non-bank").sum())
        total = len(df)
        share = (nb / total * 100) if total else 0.0
        with c4: _kpi_card("Non-bank Share", f"{share:.1f}%", f"{nb} of {total}")

    # ─────────────── COMPOSITION & RISK ───────────────
    st.markdown("## 🧭 Composition & Risk")

    # Approval vs Denial (pie)
    if "decision" in cols:
        pie_df = df["decision"].value_counts().rename_axis("Decision").reset_index(name="Count")
        fig = px.pie(pie_df, names="Decision", values="Count", title="Decision Mix")
        fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=360, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    # Avg DTI / LTV by decision (grouped bars)
    have_dti = "DTI" in cols
    have_ltv = "LTV" in cols
    if "decision" in cols and (have_dti or have_ltv):
        agg_map = {}
        if have_dti: agg_map["avg_DTI"] = ("DTI", "mean")
        if have_ltv: agg_map["avg_LTV"] = ("LTV", "mean")
        grp = df.groupby("decision").agg(**agg_map).reset_index()
        melted = grp.melt(id_vars=["decision"], var_name="metric", value_name="value")
        fig = px.bar(melted, x="decision", y="value", color="metric",
                     barmode="group", title="Average DTI / LTV by Decision")
        fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=360, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    # Loan term mix (stacked)
    term_col = "loan_term_months" if "loan_term_months" in cols else ("loan_duration_months" if "loan_duration_months" in cols else None)
    if term_col and "decision" in cols:
        mix = df.groupby([term_col, "decision"]).size().reset_index(name="count")
        fig = px.bar(
            mix, x=term_col, y="count", color="decision", title="Loan Term Mix",
            labels={term_col: "Term (months)", "count": "Count"}, barmode="stack"
        )
        fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=360, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    # Collateral avg value by type (bar)
    if {"collateral_type", "collateral_value"} <= set(cols):
        cprof = df.groupby("collateral_type").agg(
            avg_col=("collateral_value", "mean"),
            cnt=("collateral_type", "count")
        ).reset_index()
        fig = px.bar(
            cprof.sort_values("avg_col", ascending=False),
            x="collateral_type", y="avg_col",
            title=f"Avg Collateral Value by Type ({currency_symbol})",
            hover_data=["cnt"]
        )
        fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=360, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    # Top proposed plans (horizontal bar)
    if "proposed_loan_option" in cols:
        plans = df["proposed_loan_option"].dropna().astype(str)
        if len(plans) > 0:
            plan_types = []
            for s in plans:
                p = _safe_json(s)
                plan_types.append(p.get("type") if isinstance(p, dict) and "type" in p else s)
            plan_df = pd.Series(plan_types).value_counts().head(10).rename_axis("plan").reset_index(name="count")
            fig = px.bar(
                plan_df, x="count", y="plan", orientation="h",
                title="Top 10 Proposed Plans"
            )
            fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=360, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    # Customer mix table (bank vs non-bank)
    if "customer_type" in cols:
        mix = df["customer_type"].value_counts().rename_axis("Customer Type").reset_index(name="Count")
        mix["Ratio"] = (mix["Count"] / mix["Count"].sum()).round(3)
        st.markdown("### 👥 Customer Mix")
        st.dataframe(mix, use_container_width=True, height=220)

# ─────────────────────────────────────────────
# TABS
tab_gen, tab_clean, tab_run, tab_review, tab_train = st.tabs([
    "🏦 Synthetic Data Generator",
    "🧹 Anonymize & Sanitize Data",
    "🤖 Credit appraisal by AI assistant",
    "🧑‍⚖️ Human Review",
    "🔁 Training (Feedback → Retrain)"
])

# ─────────────────────────────────────────────
# DATA GENERATORS

def generate_raw_synthetic(n: int, non_bank_ratio: float) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    names = ["Alice Nguyen","Bao Tran","Chris Do","Duy Le","Emma Tran",
             "Felix Nguyen","Giang Ho","Hanh Vo","Ivan Pham","Julia Ngo"]
    emails = [f"{n.split()[0].lower()}.{n.split()[1].lower()}@gmail.com" for n in names]
    addrs = [
        "23 Elm St, Boston, MA","19 Pine Ave, San Jose, CA","14 High St, London, UK",
        "55 Nguyen Hue, Ho Chi Minh","78 Oak St, Chicago, IL","10 Broadway, New York, NY",
        "8 Rue Lafayette, Paris, FR","21 Königstr, Berlin, DE","44 Maple Dr, Los Angeles, CA","22 Bay St, Toronto, CA"
    ]
    is_non_bank = rng.random(n) < non_bank_ratio
    customer_type = np.where(is_non_bank, "non-bank", "bank")

    df = pd.DataFrame({
        "application_id": [f"APP_{i:04d}" for i in range(1, n + 1)],
        "customer_name": np.random.choice(names, n),
        "email": np.random.choice(emails, n),
        "phone": [f"+1-202-555-{1000+i:04d}" for i in range(n)],
        "address": np.random.choice(addrs, n),
        "national_id": rng.integers(10_000_000, 99_999_999, n),
        "age": rng.integers(21, 65, n),
        "income": rng.integers(25_000, 150_000, n),
        "employment_length": rng.integers(0, 30, n),
        "loan_amount": rng.integers(5_000, 100_000, n),
        "loan_duration_months": rng.choice([12, 24, 36, 48, 60, 72], n),
        "collateral_value": rng.integers(8_000, 200_000, n),
        "collateral_type": rng.choice(["real_estate","car","land","deposit"], n),
        "co_loaners": rng.choice([0,1,2], n, p=[0.7, 0.25, 0.05]),
        "credit_score": rng.integers(300, 850, n),
        "existing_debt": rng.integers(0, 50_000, n),
        "assets_owned": rng.integers(10_000, 300_000, n),
        "current_loans": rng.integers(0, 5, n),
        "customer_type": cust_type,
    })
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

    df = pd.DataFrame({
        "application_id": [f"APP_{i:04d}" for i in range(1, n + 1)],
        "age": rng.integers(21, 65, n),
        "income": rng.integers(25_000, 150_000, n),
        "employment_length": rng.integers(0, 30, n),
        "loan_amount": rng.integers(5_000, 100_000, n),
        "loan_duration_months": rng.choice([12, 24, 36, 48, 60, 72], n),
        "collateral_value": rng.integers(8_000, 200_000, n),
        "collateral_type": rng.choice(["real_estate","car","land","deposit"], n),
        "co_loaners": rng.choice([0,1,2], n, p=[0.7, 0.25, 0.05]),
        "credit_score": rng.integers(300, 850, n),
        "existing_debt": rng.integers(0, 50_000, n),
        "assets_owned": rng.integers(10_000, 300_000, n),
        "current_loans": rng.integers(0, 5, n),
        "customer_type": cust_type,
    })
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
    """
    Harmonize to the server-side agent’s expected schema.
    """
    out = df.copy()
    if "application_id" not in out.columns:
        out["application_id"] = [f"APP_{i:04d}" for i in range(1, len(out) + 1)]
    out["application_id"] = out["application_id"].astype(str)
    return out

# ────────────────────────────────
# DARK MODE STYLES
# ────────────────────────────────
st.markdown("""
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
""", unsafe_allow_html=True)

# ────────────────────────────────
# SESSION DEFAULTS
# ────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_info" not in st.session_state:
    st.session_state.user_info = {"flagged": False}
st.session_state.user_info.setdefault("flagged", False)
st.session_state.user_info.setdefault("timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
if "workflow_stage" not in st.session_state:
    st.session_state.workflow_stage = "data"
if "currency_code_label" not in st.session_state:
    st.session_state.currency_code_label = DEFAULT_CURRENCY
set_currency_defaults()

# ────────────────────────────────
# HERO / PIPELINE
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

# ────────────────────────────────
# LANDING / LOGIN
# ────────────────────────────────
def render_landing():
    col1, col2 = st.columns([1.1, 1.9], gap="large")
    with col1:
        st.markdown("<div class='left-box'>", unsafe_allow_html=True)
        logo_path = load_image("people_logo")
        if logo_path:
            st.image(logo_path, width=160)
        else:
            out["debt_to_income"] = 0.0
    rng = np.random.default_rng(12345)
    if "credit_history_length" not in out.columns:
        out["credit_history_length"] = rng.integers(0, 30, n)
    if "num_delinquencies" not in out.columns:
        out["num_delinquencies"] = np.minimum(rng.poisson(0.2, n), 10)
    if "requested_amount" not in out.columns:
        out["requested_amount"] = out.get("loan_amount", 0)
    if "loan_term_months" not in out.columns:
        out["loan_term_months"] = out.get("loan_duration_months", 0)
    return dedupe_columns(out)

# ─────────────────────────────────────────────
# 🏦 TAB 1 — Synthetic Data Generator
with tab_gen:
    st.subheader("🏦 Synthetic Credit Data Generator")

    # Currency selector (before generation)
    c1, c2 = st.columns([1, 2])
    with c1:
        code = st.selectbox(
            "Currency",
            list(CURRENCY_OPTIONS.keys()),
            index=list(CURRENCY_OPTIONS.keys()).index(st.session_state["currency_code"]),
            help="All monetary fields will be in this local currency."
        )
        if code != st.session_state["currency_code"]:
            st.session_state["currency_code"] = code
            set_currency_defaults()
    with c2:
        st.info(f"Amounts will be generated in **{st.session_state['currency_label']}**.", icon="💰")

    rows = st.slider("Number of rows to generate", 50, 2000, 200, step=50)
    non_bank_ratio = st.slider("Share of non-bank customers", 0.0, 1.0, 0.30, 0.05)

    col_raw, col_anon = st.columns(2)
    with col_raw:
        if st.button("🔴 Generate RAW Synthetic Data (with PII)", use_container_width=True):
            raw_df = generate_raw_synthetic(rows, non_bank_ratio)
            st.session_state.synthetic_raw_df = raw_df
            raw_path = save_to_runs(raw_df, "synthetic_raw")
            st.success(f"Generated RAW (PII) dataset with {rows} rows in {st.session_state['currency_label']}. Saved to {raw_path}")
            st.dataframe(raw_df.head(10), use_container_width=True)
            st.download_button(
                "⬇️ Download RAW CSV",
                raw_df.to_csv(index=False).encode("utf-8"),
                os.path.basename(raw_path),
                "text/csv"
            )

    with colB:
        if st.button("🟢 Generate ANON Synthetic Data (ready for agent)", use_container_width=True):
            anon_df = generate_anon_synthetic(rows, non_bank_ratio)
            st.session_state.synthetic_df = anon_df
            anon_path = save_to_runs(anon_df, "synthetic_anon")
            st.success(f"Generated ANON dataset with {rows} rows in {st.session_state['currency_label']}. Saved to {anon_path}")
            st.dataframe(anon_df.head(10), use_container_width=True)
            st.download_button(
                "⬇️ Download ANON CSV",
                anon_df.to_csv(index=False).encode("utf-8"),
                os.path.basename(anon_path),
                "text/csv"
            )

# ─────────────────────────────────────────────
# 🧹 TAB 2 — Anonymize & Sanitize Data
with tab_clean:
    st.subheader("🧹 Upload & Anonymize Customer Data (PII columns will be DROPPED)")
    st.markdown("Upload your **real CSV**. We drop PII columns and scrub emails/phones in text fields.")

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()
        st.write("📊 Original Data Preview:")
        st.dataframe(dedupe_columns(df.head(5)), use_container_width=True)

        sanitized, dropped_cols = drop_pii_columns(df)
        sanitized = append_user_info(sanitized)
        sanitized = dedupe_columns(sanitized)
        st.session_state.anonymized_df = sanitized
        st.success(f"Dropped possible PII columns: {', '.join(dropped_cols) if dropped_cols else 'none'}")
        st.dataframe(sanitized.head(10), use_container_width=True)

    nav = st.columns([1,1,1])
    with nav[0]:
        if st.button("➡️ Continue to KYC"):
            st.session_state.workflow_stage = "kyc"
            st.rerun()

def page_kyc():
    render_pipeline_hero("kyc")
    st.title("🛂 KYC & Compliance Workbench")
    st.caption("Capture applicant identity, perform sanctions checks, and feed compliance context downstream.")

    st.session_state.user_info.setdefault("timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if st.button("🔁 Refresh Synthetic KYC Dossier"):
        build_session_kyc_registry(force=True)
        st.success("Synthetic KYC dossier refreshed.")
        st.rerun()

        st.success(f"Dropped PII columns: {sorted(dropped_cols) if dropped_cols else 'None'}")
        st.write("✅ Sanitized Data Preview:")
        st.dataframe(sanitized.head(5), use_container_width=True)

        fpath = save_to_runs(sanitized, "anonymized")
        st.success(f"Saved anonymized file: {fpath}")
        st.download_button(
            "⬇️ Download Clean Data",
            sanitized.to_csv(index=False).encode("utf-8"),
            os.path.basename(fpath),
            "text/csv"
        )
    else:
        st.info("Choose a CSV to see the sanitize flow.", icon="ℹ️")

# ─────────────────────────────────────────────
# 🤖 TAB 3 — Credit appraisal by AI assistant
with tab_run:
    st.subheader("🤖 Credit appraisal by AI assistant")

    # Production model banner (optional)
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

    # 1) Model + Hardware selection (UI hints)
    LLM_MODELS = [
        ("Phi-3 Mini (3.8B) — CPU OK", "phi3:3.8b", "CPU 8GB RAM (fast)"),
        ("Mistral 7B Instruct — CPU slow / GPU OK", "mistral:7b-instruct", "CPU 16GB (slow) or GPU ≥8GB"),
        ("Gemma-2 7B — CPU slow / GPU OK", "gemma2:7b", "CPU 16GB (slow) or GPU ≥8GB"),
        ("LLaMA-3 8B — GPU recommended", "llama3:8b-instruct", "GPU ≥12GB (CPU very slow)"),
        ("Qwen2 7B — GPU recommended", "qwen2:7b-instruct", "GPU ≥12GB (CPU very slow)"),
        ("Mixtral 8x7B — GPU only (big)", "mixtral:8x7b-instruct", "GPU 24–48GB"),
    ]
    LLM_LABELS = [l for (l, _, _) in LLM_MODELS]
    LLM_VALUE_BY_LABEL = {l: v for (l, v, _) in LLM_MODELS}
    LLM_HINT_BY_LABEL = {l: h for (l, _, h) in LLM_MODELS}

    OPENSTACK_FLAVORS = {
        "m4.medium":  "4 vCPU / 8 GB RAM — CPU-only small",
        "m8.large":   "8 vCPU / 16 GB RAM — CPU-only medium",
        "g1.a10.1":   "8 vCPU / 32 GB RAM + 1×A10 24GB",
        "g1.l40.1":   "16 vCPU / 64 GB RAM + 1×L40 48GB",
        "g2.a100.1":  "24 vCPU / 128 GB RAM + 1×A100 80GB",
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

    # 2) Data Source
    data_choice = st.selectbox(
        "Select Data Source",
        [
            "Use synthetic (ANON)",
            "Use synthetic (RAW – auto-sanitize)",
            "Use anonymized dataset",
            "Upload manually",
        ]
    )
    use_llm = st.checkbox("Use LLM narrative", value=False)
    agent_name = "credit_appraisal"

    if data_choice == "Upload manually":
        up = st.file_uploader("Upload your CSV", type=["csv"], key="manual_upload_run_file")
        if up is not None:
            st.session_state["manual_upload_name"] = up.name
            st.session_state["manual_upload_bytes"] = up.getvalue()
            st.success(f"File staged: {up.name} ({len(st.session_state['manual_upload_bytes'])} bytes)")

    # 3) Rules
    st.markdown("### ⚙️ Decision Rule Set")
    rule_mode = st.radio(
        "Choose rule mode",
        ["Classic (bank-style metrics)", "NDI (Net Disposable Income) — simple"],
        index=0,
        help="NDI = income - all monthly obligations. Approve if NDI and NDI ratio pass thresholds."
    )

    CLASSIC_DEFAULTS = {
        "max_dti": 0.45, "min_emp_years": 2, "min_credit_hist": 3, "salary_floor": 3000,
        "max_delinquencies": 2, "max_current_loans": 3, "req_min": 1000, "req_max": 200000,
        "loan_terms": [12, 24, 36, 48, 60], "threshold": 0.45, "target_rate": None, "random_band": True,
        "min_income_debt_ratio": 0.35, "compounded_debt_factor": 1.0, "monthly_debt_relief": 0.50,
    }
    NDI_DEFAULTS = {"ndi_value": 800.0, "ndi_ratio": 0.50, "threshold": 0.45, "target_rate": None, "random_band": True}

    if "classic_rules" not in st.session_state:
        st.session_state.classic_rules = CLASSIC_DEFAULTS.copy()
    if "ndi_rules" not in st.session_state:
        st.session_state.ndi_rules = NDI_DEFAULTS.copy()

    def reset_classic(): st.session_state.classic_rules = CLASSIC_DEFAULTS.copy()
    def reset_ndi():     st.session_state.ndi_rules = NDI_DEFAULTS.copy()

    if rule_mode.startswith("Classic"):
        with st.expander("Classic Metrics (with Reset)", expanded=True):
            rc = st.session_state.classic_rules
            r1, r2, r3 = st.columns(3)
            with r1:
                rc["max_dti"] = st.slider("Max Debt-to-Income (DTI)", 0.0, 1.0, rc["max_dti"], 0.01)
                rc["min_emp_years"] = st.number_input("Min Employment Years", 0, 40, rc["min_emp_years"])
                rc["min_credit_hist"] = st.number_input("Min Credit History (years)", 0, 40, rc["min_credit_hist"])
            with r2:
                rc["salary_floor"] = st.number_input("Minimum Monthly Salary", 0, 1_000_000_000, rc["salary_floor"], step=1000, help=fmt_currency_label("in local currency"))
                rc["max_delinquencies"] = st.number_input("Max Delinquencies", 0, 10, rc["max_delinquencies"])
                rc["max_current_loans"] = st.number_input("Max Current Loans", 0, 10, rc["max_current_loans"])
            with r3:
                rc["req_min"] = st.number_input(fmt_currency_label("Requested Amount Min"), 0, 10_000_000_000, rc["req_min"], step=1000)
                rc["req_max"] = st.number_input(fmt_currency_label("Requested Amount Max"), 0, 10_000_000_000, rc["req_max"], step=1000)
                rc["loan_terms"] = st.multiselect("Allowed Loan Terms (months)", [12,24,36,48,60,72], default=rc["loan_terms"])

            st.markdown("#### 🧮 Debt Pressure Controls")
            d1, d2, d3 = st.columns(3)
            with d1:
                rc["min_income_debt_ratio"] = st.slider("Min Income / (Compounded Debt) Ratio", 0.10, 2.00, rc["min_income_debt_ratio"], 0.01)
            with d2:
                rc["compounded_debt_factor"] = st.slider("Compounded Debt Factor (× requested)", 0.5, 3.0, rc["compounded_debt_factor"], 0.1)
            with d3:
                rc["monthly_debt_relief"] = st.slider("Monthly Debt Relief Factor", 0.10, 1.00, rc["monthly_debt_relief"], 0.05)

            st.markdown("---")
            c1, c2, c3 = st.columns([1,1,1])
            with c1:
                use_target = st.toggle("🎯 Use target approval rate", value=(rc["target_rate"] is not None))
            with c2:
                rc["random_band"] = st.toggle("🎲 Randomize approval band (20–60%) when no target", value=rc["random_band"])
            with c3:
                if st.button("↩️ Reset to defaults"):
                    reset_classic()
                    st.rerun()

            if use_target:
                rc["target_rate"] = st.slider("Target approval rate", 0.05, 0.95, rc["target_rate"] or 0.40, 0.01)
                rc["threshold"] = None
            else:
                rc["threshold"] = st.slider("Model score threshold", 0.0, 1.0, rc["threshold"], 0.01)
                rc["target_rate"] = None
    else:
        with st.expander("NDI Metrics (with Reset)", expanded=True):
            rn = st.session_state.ndi_rules
            n1, n2 = st.columns(2)
            with n1:
                rn["ndi_value"] = st.number_input(fmt_currency_label("Min NDI (Net Disposable Income) per month"), 0.0, 1e12, float(rn["ndi_value"]), step=50.0)
            with n2:
                rn["ndi_ratio"] = st.slider("Min NDI / Income ratio", 0.0, 1.0, float(rn["ndi_ratio"]), 0.01)
            st.caption("NDI = income - all monthly obligations (rent, food, loans, cards, etc.).")

            st.markdown("---")
            c1, c2, c3 = st.columns([1,1,1])
            with c1:
                use_target = st.toggle("🎯 Use target approval rate", value=(rn["target_rate"] is not None))
            with c2:
                rn["random_band"] = st.toggle("🎲 Randomize approval band (20–60%) when no target", value=rn["random_band"])
            with c3:
                if st.button("↩️ Reset to defaults (NDI)"):
                    reset_ndi()
                    st.rerun()

            if use_target:
                rn["target_rate"] = st.slider("Target approval rate", 0.05, 0.95, rn["target_rate"] or 0.40, 0.01)
                rn["threshold"] = None
            else:
                rn["threshold"] = st.slider("Model score threshold", 0.0, 1.0, rn["threshold"], 0.01)
                rn["target_rate"] = None

    # 4) Run
    if st.button("🚀 Run Agent", use_container_width=True):
        try:
            files = None
            data: Dict[str, Any] = {
                "use_llm_narrative": str(use_llm).lower(),
                "llm_model": llm_value,
                "hardware_flavor": flavor,
                "currency_code": st.session_state["currency_code"],
                "currency_symbol": st.session_state["currency_symbol"],
            }
            if rule_mode.startswith("Classic"):
                rc = st.session_state.classic_rules
                data.update({
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
                })
            else:
                rn = st.session_state.ndi_rules
                data.update({
                    "ndi_value": str(rn["ndi_value"]),
                    "ndi_ratio": str(rn["ndi_ratio"]),
                    "threshold": "" if rn["threshold"] is None else str(rn["threshold"]),
                    "target_approval_rate": "" if rn["target_rate"] is None else str(rn["target_rate"]),
                    "random_band": str(rn["random_band"]).lower(),
                    "random_approval_band": str(rn["random_band"]).lower(),
                    "rule_mode": "ndi",
                })

            def prep_and_pack(df: pd.DataFrame, filename: str):
                safe = dedupe_columns(df)
                safe, _ = drop_pii_columns(safe)
                safe = strip_policy_banned(safe)
                safe = to_agent_schema(safe)
                buf = io.StringIO()
                safe.to_csv(buf, index=False)
                return {"file": (filename, buf.getvalue().encode("utf-8"), "text/csv")}

            if data_choice == "Use synthetic (ANON)":
                if "synthetic_df" not in st.session_state:
                    st.warning("No ANON synthetic dataset found. Generate it in the first tab."); st.stop()
                files = prep_and_pack(st.session_state.synthetic_df, "synthetic_anon.csv")

            elif data_choice == "Use synthetic (RAW – auto-sanitize)":
                if "synthetic_raw_df" not in st.session_state:
                    st.warning("No RAW synthetic dataset found. Generate it in the first tab."); st.stop()
                files = prep_and_pack(st.session_state.synthetic_raw_df, "synthetic_raw_sanitized.csv")

            elif data_choice == "Use anonymized dataset":
                if "anonymized_df" not in st.session_state:
                    st.warning("No anonymized dataset found. Create it in the second tab."); st.stop()
                files = prep_and_pack(st.session_state.anonymized_df, "anonymized.csv")

            elif data_choice == "Upload manually":
                up_name = st.session_state.get("manual_upload_name")
                up_bytes = st.session_state.get("manual_upload_bytes")
                if not up_name or not up_bytes:
                    st.warning("Please upload a CSV first."); st.stop()
                try:
                    tmp_df = pd.read_csv(io.BytesIO(up_bytes))
                    files = prep_and_pack(tmp_df, up_name)
                except Exception:
                    files = {"file": (up_name, up_bytes, "text/csv")}
            else:
                st.error("Unknown data source selection."); st.stop()

            r = requests.post(f"{API_URL}/v1/agents/{agent_name}/run", data=data, files=files, timeout=180)
            if r.status_code != 200:
                st.error(f"Run failed ({r.status_code}): {r.text}"); st.stop()

            res = r.json()
            st.session_state.last_run_id = res.get("run_id")
            result = res.get("result", {}) or {}
            st.success(f"✅ Run succeeded! Run ID: {st.session_state.last_run_id}")

            # Pull merged.csv for dashboards/review
            rid = st.session_state.last_run_id
            merged_url = f"{API_URL}/v1/runs/{rid}/report?format=csv"
            merged_bytes = requests.get(merged_url, timeout=30).content
            merged_df = pd.read_csv(io.BytesIO(merged_bytes))
            st.session_state["last_merged_df"] = merged_df

            # Export AI outputs as csv with currency code (for Human Review dropdown)
            ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            out_name = f"ai-appraisal-outputs-{ts}-{st.session_state['currency_code']}.csv"
            st.download_button("⬇️ Download AI outputs (CSV)", merged_df.to_csv(index=False).encode("utf-8"), out_name, "text/csv")

            # Decision filter IN TABLE (not hiding dashboard)
            st.markdown("### 📄 Credit Ai Agent  Decisions Table (filtered)")
            uniq_dec = sorted([d for d in merged_df.get("decision", pd.Series(dtype=str)).dropna().unique()])
            chosen = st.multiselect("Filter decision", options=uniq_dec, default=uniq_dec, key="filter_decisions")
            df_view = merged_df.copy()
            if "decision" in df_view.columns and chosen:
                df_view = df_view[df_view["decision"].isin(chosen)]
            st.dataframe(df_view, use_container_width=True)

            # ── DASHBOARD (always visible; filters apply in table below)
            st.markdown("## 📊 Dashboard")
            render_credit_dashboard(merged_df, st.session_state.get("currency_symbol", ""))

            # Per-row metrics met/not met
            if "rule_reasons" in df_view.columns:
                rr = df_view["rule_reasons"].apply(try_json)
                df_view["metrics_met"] = rr.apply(lambda d: ", ".join(sorted([k for k, v in (d or {}).items() if v is True])) if isinstance(d, dict) else "")
                df_view["metrics_unmet"] = rr.apply(lambda d: ", ".join(sorted([k for k, v in (d or {}).items() if v is False])) if isinstance(d, dict) else "")
            cols_show = [c for c in [
                "application_id","customer_type","decision","score","loan_amount","income","metrics_met","metrics_unmet",
                "proposed_loan_option","proposed_consolidation_loan","top_feature","explanation"
            ] if c in df_view.columns]
            st.dataframe(df_view[cols_show].head(500), use_container_width=True)

            # Downloads
            cdl1, cdl2, cdl3, cdl4, cdl5 = st.columns(5)
            with cdl1: st.markdown(f"[⬇️ PDF report]({API_URL}/v1/runs/{rid}/report?format=pdf)")
            with cdl2: st.markdown(f"[⬇️ Scores CSV]({API_URL}/v1/runs/{rid}/report?format=scores_csv)")
            with cdl3: st.markdown(f"[⬇️ Explanations CSV]({API_URL}/v1/runs/{rid}/report?format=explanations_csv)")
            with cdl4: st.markdown(f"[⬇️ Merged CSV]({API_URL}/v1/runs/{rid}/report?format=csv)")
            with cdl5: st.markdown(f"[⬇️ JSON]({API_URL}/v1/runs/{rid}/report?format=json)")

        except Exception as e:
            st.exception(e)

    # Re-download quick section
    if st.session_state.get("last_run_id"):
        st.markdown("---")
        st.subheader("📥 Download Latest Outputs")
        rid = st.session_state.last_run_id
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1: st.markdown(f"[⬇️ PDF]({API_URL}/v1/runs/{rid}/report?format=pdf)")
        with col2: st.markdown(f"[⬇️ Scores CSV]({API_URL}/v1/runs/{rid}/report?format=scores_csv)")
        with col3: st.markdown(f"[⬇️ Explanations CSV]({API_URL}/v1/runs/{rid}/report?format=explanations_csv)")
        with col4: st.markdown(f"[⬇️ Merged CSV]({API_URL}/v1/runs/{rid}/report?format=csv)")
        with col5: st.markdown(f"[⬇️ JSON]({API_URL}/v1/runs/{rid}/report?format=json)")


# ─────────────────────────────────────────────
# 🧑‍⚖️ TAB 4 — Human Review
with tab_review:
    st.subheader("🧑‍⚖️ Human Review — Correct AI Decisions & Score Agreement")

    # Allow loading AI output CSV back into review via dropdown upload
    uploaded_review = st.file_uploader("Load AI outputs CSV for review (optional)", type=["csv"], key="review_csv_loader")
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
    editable = st.data_editor(editable, num_rows="dynamic", use_container_width=True, key="review_editor")

    if st.button("💾 Save corrections"):
        st.session_state["review_corrections"] = editable
        st.success("Corrections saved in session.")

    nav = st.columns([1,1,1])
    with nav[0]:
        if st.button("⬅️ Back to Credit"):
            st.session_state.workflow_stage = "credit"
            st.rerun()
    with nav[1]:
        if st.button("➡️ Continue to Training"):
            st.session_state.workflow_stage = "training"
            st.rerun()

def page_training():
    render_pipeline_hero("training")
    st.title("🔁 Training (Feedback → Retrain)")
    st.caption("Loop curated feedback into retraining jobs and promote production-ready models.")

    corr = st.session_state.get("review_corrections")
    if corr is not None:
        st.write("Recent corrections:")
        st.dataframe(corr, use_container_width=True)
    else:
        dfm = st.session_state["last_merged_df"].copy()
        st.markdown("#### 1) Select rows to review and correct")

        editable_cols = []
        if "decision" in dfm.columns: editable_cols.append("decision")
        if "rule_reasons" in dfm.columns: editable_cols.append("rule_reasons")
        if "customer_type" in dfm.columns: editable_cols.append("customer_type")

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
                "customer_type": st.column_config.SelectboxColumn(options=["bank", "non-bank"], disabled=True)
            }
        )

    if st.button("📦 Export training dataset (CSV)"):
        # For now, export latest run merged with corrections if any
        base = st.session_state.get("last_merged_df")
        if base is None or base.empty:
            st.warning("No base run to export.")
        else:
            out = base.copy()
            if corr is not None and "application_id" in corr.columns and "human_decision" in corr.columns:
                out = out.merge(corr[["application_id", "human_decision"]], on="application_id", how="left")
            buf = io.StringIO()
            out.to_csv(buf, index=False)
            st.download_button("⬇️ Download Training CSV", buf.getvalue().encode("utf-8"), "training_dataset.csv", "text/csv")

    nav = st.columns([1,1,1])
    with nav[0]:
        if st.button("⬅️ Back to Review"):
            st.session_state.workflow_stage = "review"
            st.rerun()

        # Export review CSV (manual loop into training)
        st.markdown("#### 3) Export review CSV")
        model_used = "production"  # if you track specific model names, set it here
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_user = st.session_state["user_info"]["name"].replace(" ", "").lower()
        review_name = f"creditappraisal.{safe_user}.{model_used}.{ts}.csv"
        csv_bytes = edited.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export review CSV", csv_bytes, review_name, "text/csv")
        st.caption(f"Saved file name pattern: **{review_name}**")


# ─────────────────────────────────────────────
# 🔁 TAB 5 — Training (Feedback → Retrain)
with tab_train:
    st.subheader("🔁 Human Feedback → Retrain (new payload)")

    st.markdown("**Drag & drop** one or more review CSVs exported from the Human Review tab.")
    up_list = st.file_uploader("Upload feedback CSV(s)", type=["csv"], accept_multiple_files=True, key="train_feedback_uploader")

    staged_paths: List[str] = []
    if up_list:
        for up in up_list:
            # stage to tmp_feedback dir
            dest = os.path.join(TMP_FEEDBACK_DIR, up.name)
            with open(dest, "wb") as f:
                f.write(up.getvalue())
            staged_paths.append(dest)
        st.success(f"Staged {len(staged_paths)} feedback file(s) to {TMP_FEEDBACK_DIR}")
        st.write(staged_paths)

    st.markdown("#### Launch Retrain")
    payload = {
        "feedback_csvs": staged_paths,
        "user_name": st.session_state["user_info"]["name"],
        "agent_name": "credit_appraisal",
        "algo_name": "credit_lr",
    }
    st.code(json.dumps(payload, indent=2), language="json")

    colA, colB = st.columns([1,1])
    with colA:
        if st.button("🚀 Train candidate model"):
            try:
                r = requests.post(f"{API_URL}/v1/training/train", json=payload, timeout=90)
                if r.ok:
                    st.success(r.json())
                    st.session_state["last_train_job"] = r.json().get("job_id")
                else:
                    st.error(r.text)
            except Exception as e:
                st.error(f"Train failed: {e}")
    with colB:
        if st.button("⬆️ Promote last candidate to PRODUCTION"):
            try:
                r = requests.post(f"{API_URL}/v1/training/promote", timeout=30)
                st.write(r.json() if r.ok else r.text)
            except Exception as e:
                st.error(f"Promote failed: {e}")

    if not st.session_state.logged_in:
        render_landing()
        st.stop()

    # Header & user flash
    flash_user = st.session_state.pop("login_flash", None)
    if flash_user:
        st.success(f"✅ Logged in as {flash_user}")

    # Main nav by workflow stage
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
    else:
        st.session_state.workflow_stage = "data"
        page_data()

if __name__ == "__main__":
    main()
