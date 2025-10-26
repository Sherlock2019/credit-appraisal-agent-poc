# services/ui/app.py
# ─────────────────────────────────────────────
# 🌐 OpenSource AI Agent Library + Credit Appraisal PoC by Dzoan
# ─────────────────────────────────────────────
from __future__ import annotations
import os
import re
import io
import json
import random
import datetime
from typing import Optional, Dict, List, Any

import pandas as pd
import numpy as np
import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go


# ────────────────────────────────
# CONFIG
# ────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8090")
RUNS_DIR = os.path.expanduser("~/credit-appraisal-agent-poc/services/api/.runs")
TMP_FEEDBACK_DIR = os.path.join(RUNS_DIR, "tmp_feedback")
LANDING_IMG_DIR = os.path.expanduser("~/credit-appraisal-agent-poc/services/ui/landing_images")

os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(TMP_FEEDBACK_DIR, exist_ok=True)
os.makedirs(LANDING_IMG_DIR, exist_ok=True)

st.set_page_config(page_title="AI Agent Sandbox — By the People, For the People", layout="wide")

# ────────────────────────────────
# SESSION DEFAULTS
# ────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_info" not in st.session_state:
    st.session_state.user_info = {}

st.session_state.user_info.setdefault("flagged", False)
st.session_state.user_info.setdefault(
    "timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
    with open(dest, "wb") as f:
        f.write(uploaded_file.getvalue())
    return dest

def render_image_tag(agent_id: str, industry: str, emoji_fallback: str) -> str:
    base = agent_id.lower().replace(" ", "_")
    img_path = load_image(base) or load_image(industry.replace(" ", "_"))
    if img_path:
        return f'<img src="file://{img_path}" style="width:48px;height:48px;border-radius:10px;object-fit:cover;">'
    else:
        return f'<div style="font-size:32px;">{emoji_fallback}</div>'

# ────────────────────────────────
# DATA
# ────────────────────────────────
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
button {
    transition: all 0.25s ease-in-out;
}
button:hover {
    transform: translateY(-2px);
    background: linear-gradient(90deg,#1d4ed8,#2563eb);
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
# LAYOUT
# ────────────────────────────────
if not st.session_state.logged_in:
    col1, col2 = st.columns([1.1, 1.9], gap="large")

    with col1:
        st.markdown("<div class='left-box'>", unsafe_allow_html=True)
        logo_path = load_image("people_logo")
        if logo_path:
            st.image(logo_path, width=160)
        else:
            logo_upload = st.file_uploader("Upload People Logo", type=["jpg", "png", "webp"], key="upload_logo")
            if logo_upload:
                save_uploaded_image(logo_upload, "people_logo")
                st.success("✅ Logo uploaded successfully! Refreshing...")
                st.rerun()

        st.markdown("""
        <h1>✊ Let’s Build an AI by the People, for the People</h1>
        <h3>⚙️ Ready-to-Use AI Agent Sandbox — From Sandbox to Production</h3>
        <p>
        A world-class open innovation space where anyone can build, test, and deploy AI agents using open-source code, explainable models, and modular templates.<br><br>
        For developers, startups, and enterprises — experiment, customize, and scale AI without barriers.<br><br>
        <b>Privacy & Data Sovereignty:</b> Each agent runs under strict privacy controls and complies with GDPR & Vietnam Data Law 2025. Only anonymized or synthetic data is used — your data never leaves your environment.<br><br>
        <b>From Sandbox to Production:</b> Start with ready-to-use agent templates, adapt, test, and deploy — all on GPU-as-a-Service Cloud with zero CAPEX.<br><br>
        You dream it — now you can build it.
        </p>
        <div style="text-align:center;margin-top:2rem;">
            <a href="#credit_poc" style="text-decoration:none;">
                <button style="background:linear-gradient(90deg,#2563eb,#1d4ed8);
                               border:none;border-radius:12px;color:white;
                               padding:16px 32px;font-size:18px;cursor:pointer;">
                    🚀 Start Building Now
                </button>
            </a>
        </div>
        """, unsafe_allow_html=True)
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
            rows.append({
                "🖼️": image_html,
                "🏭 Sector": sector,
                "🧩 Industry": industry,
                "🤖 Agent": agent,
                "🧠 Description": desc,
                "📶 Status": f'<span class="status-{status.replace(" ", "")}">{status}</span>',
                "⭐ Rating": "⭐" * int(rating) + "☆" * (5 - int(rating)),
                "👥 Users": users,
                "💬 Comments": comments
            })
        df = pd.DataFrame(rows)
        st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # WORKFLOW PIPELINE — WITH LOOPBACK
    # ─────────────────────────────────────────────
    st.markdown(
        """
        ### 🛠️ Workflow Pipeline Overview
        1. **Synthetic Data Generator** – Create realistic datasets for testing.
        2. **Anonymize & Sanitize Data** – Drop PII and scrub sensitive text.
        3. **Credit Appraisal by AI Assistant** – Run agent-driven credit decisions.
        4. **Human Review** – Evaluate and adjust AI outputs.
        5. **Training (Feedback → Retrain)** – Feed human-labelled data back into training.
        6. **Loop Back** – Re-run the agent with the newly trained model.
        """
    )

    # ────────────────────────────────
    # FOOTER
    # ────────────────────────────────
    st.markdown("<footer>Made with ❤️ by Dzoan Nguyen— Open AI Sandbox Initiative</footer>", unsafe_allow_html=True)

    # ── Login Screen
    with st.container():
        st.markdown("### 🔐 Login (Demo Mode)")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            username = st.text_input("Username", value="", placeholder="e.g. dzoan")
        with col2:
            email = st.text_input("Email", value="", placeholder="e.g. dzoan@demo.local")
        with col3:
            password = st.text_input("Password", type="password", placeholder="Enter any password")

        login_btn = st.button("Login", type="primary", use_container_width=True)

        if login_btn:
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
                st.session_state["login_flash"] = username.strip()
            else:
                st.error("Please enter both username and email to continue.")

    st.stop()


# ─────────────────────────────────────────────
# HEADER — USER INFO + SECURITY

flash_user = st.session_state.pop("login_flash", None)
if flash_user:
    st.success(f"✅ Logged in as {flash_user}")

st.title("💳 AI Credit Appraisal Platform")
st.caption("Generate, sanitize, and appraise credit with AI agent Power and Human Decisions  .")

# Short aliases for backward compatibility
user_name = st.session_state.user_info.get("name", "")
user_email = st.session_state.user_info.get("email", "")
flag_session = st.session_state.user_info.get("flagged", False)


# ─────────────────────────────────────────────
# GLOBAL UTILS

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
    if "user_info" not in st.session_state:
        st.session_state.user_info = {}
    meta = st.session_state.user_info
    timestamp = meta.get("timestamp") or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta["timestamp"] = timestamp

    out = df.copy()
    out["session_user_name"] = meta.get("name", "")
    out["session_user_email"] = meta.get("email", "")
    out["session_flagged"] = meta.get("flagged", False)
    out["created_at"] = timestamp
    return dedupe_columns(out)

def save_to_runs(df: pd.DataFrame, prefix: str) -> str:
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    flagged = bool(st.session_state.get("user_info", {}).get("flagged", False))
    flag_suffix = "_FLAGGED" if flagged else ""
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
        return json.loads(x)
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
        st.info("No data to visualize yet.")
        return

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
    is_non = rng.random(n) < non_bank_ratio
    cust_type = np.where(is_non, "non-bank", "bank")

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
    df["CWI"] = ((1 - df["DTI"]).clip(0, 1)) * ((1 - df["LTV"]).clip(0, 1)) * (df["CCR"].clip(0, 3))

    fx = st.session_state["currency_fx"]
    for c in ("income", "loan_amount", "collateral_value", "assets_owned", "existing_debt"):
        df[c] = (df[c] * fx).round(2)
    df["currency_code"] = st.session_state["currency_code"]
    return dedupe_columns(df)

def generate_anon_synthetic(n: int, non_bank_ratio: float) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    is_non = rng.random(n) < non_bank_ratio
    cust_type = np.where(is_non, "non-bank", "bank")

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
    df["CWI"] = ((1 - df["DTI"]).clip(0, 1)) * ((1 - df["LTV"]).clip(0, 1)) * (df["CCR"].clip(0, 3))

    fx = st.session_state["currency_fx"]
    for c in ("income", "loan_amount", "collateral_value", "assets_owned", "existing_debt"):
        df[c] = (df[c] * fx).round(2)
    df["currency_code"] = st.session_state["currency_code"]
    return dedupe_columns(df)

def to_agent_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonize to the server-side agent’s expected schema.
    """
    out = df.copy()
    n = len(out)
    if "employment_years" not in out.columns:
        out["employment_years"] = out.get("employment_length", 0)
    if "debt_to_income" not in out.columns:
        if "DTI" in out.columns:
            out["debt_to_income"] = out["DTI"].astype(float)
        elif "existing_debt" in out.columns and "income" in out.columns:
            denom = out["income"].replace(0, np.nan)
            dti = (out["existing_debt"] / denom).fillna(0.0)
            out["debt_to_income"] = dti.clip(0, 10)
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

    colA, colB = st.columns(2)
    with colA:
        if st.button("🔴 Generate RAW Synthetic Data (with PII)", use_container_width=True):
            raw_df = append_user_info(generate_raw_synthetic(rows, non_bank_ratio))
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
            anon_df = append_user_info(generate_anon_synthetic(rows, non_bank_ratio))
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

    if "asset_appraisal_result" not in st.session_state:
        st.session_state["asset_appraisal_result"] = None
    if "asset_verified_result" not in st.session_state:
        st.session_state["asset_verified_result"] = None

    st.markdown("### 🏠 Collateral Asset Workflow")
    latest_asset_context = st.session_state.get("asset_verified_result") or st.session_state.get("asset_appraisal_result")
    default_asset_id = (latest_asset_context or {}).get("asset_id", "")
    if latest_asset_context:
        try:
            est_value = float(latest_asset_context.get("estimated_value", 0) or 0.0)
            confidence_pct = float(latest_asset_context.get("confidence", 0) or 0.0) * 100.0
        except Exception:
            est_value = 0.0
            confidence_pct = 0.0
        st.info(
            f"Latest asset valuation: **{latest_asset_context.get('asset_type', 'Asset')}** — "
            f"ID `{default_asset_id}` — Estimated value ${est_value:,.0f} "
            f"(confidence {confidence_pct:.1f}% ).",
            icon="🏠",
        )

    with st.expander("🏠 Asset Appraisal Agent", expanded=False):
        asset_type = st.selectbox(
            "Asset Type",
            ["Real Estate", "Vehicle", "Equipment", "Other"],
            key="asset_type_selector",
        )
        declared_value = st.number_input(
            "Declared Asset Value",
            min_value=0.0,
            step=1000.0,
            help="Optional declared value from borrower.",
        )
        asset_notes = st.text_area(
            "Asset Notes",
            "",
            help="Add contextual notes for the asset appraisal.",
        )
        uploaded_docs = st.file_uploader(
            "Upload Documents / Photos",
            accept_multiple_files=True,
            key="asset_docs_uploader",
        )
        metadata: Dict[str, Any] = {"notes": asset_notes}
        if declared_value:
            metadata["declared_value"] = declared_value
        if uploaded_docs:
            metadata["files"] = [doc.name for doc in uploaded_docs]

        if st.button("Run Asset Appraisal", key="asset_appraisal_run_btn"):
            try:
                payload = {
                    "asset_type": asset_type,
                    "metadata": json.dumps({k: v for k, v in metadata.items() if v not in (None, "", [])}),
                }
                resp = requests.post(
                    f"{API_URL}/v1/agents/asset_appraisal/run",
                    data=payload,
                    timeout=30,
                )
                resp.raise_for_status()
                result = resp.json().get("result")
                st.session_state["asset_appraisal_result"] = result
                st.session_state["asset_verified_result"] = None
                st.success("Asset appraisal completed.")
                st.json(result)
                default_asset_id = result.get("asset_id", "") if result else ""
                latest_asset_context = result
            except Exception as exc:
                st.error(f"Asset appraisal failed: {exc}")

    with st.expander("🕵️ Bank Inspector Verification", expanded=False):
        verify_asset_id = st.text_input(
            "Asset ID",
            value=default_asset_id,
            key="asset_verify_id",
        )
        verified_flag = st.checkbox(
            "Legitimacy Verified",
            value=True,
            key="asset_verify_flag",
        )
        legitimacy_score = st.slider(
            "Legitimacy Score",
            0.0,
            1.0,
            0.95,
            0.01,
        )
        inspector_notes = st.text_area(
            "Inspector Notes",
            "Land title confirmed by local authority.",
        )
        local_ref = st.text_input("Local Authority Reference", "")

        if st.button("Submit Verification", key="asset_submit_verification"):
            try:
                data = {
                    "asset_id": verify_asset_id,
                    "verified": str(verified_flag).lower(),
                    "legitimacy_score": legitimacy_score,
                    "inspector_notes": inspector_notes,
                    "local_authority_ref": local_ref,
                }
                resp = requests.post(
                    f"{API_URL}/v1/agents/asset_appraisal/verify",
                    data=data,
                    timeout=30,
                )
                resp.raise_for_status()
                result = resp.json().get("result")
                st.session_state["asset_verified_result"] = result
                st.success("Verification recorded and asset valuation updated.")
                st.json(result)
                latest_asset_context = result
                default_asset_id = result.get("asset_id", verify_asset_id) if result else verify_asset_id
            except Exception as exc:
                st.error(f"Failed to submit verification: {exc}")

    with st.expander("📱 Field Data Upload", expanded=False):
        field_asset_id = st.text_input(
            "Asset ID",
            value=default_asset_id,
            key="field_asset_id",
        )
        inspector_name = st.text_input(
            "Inspector Name",
            value="",
            key="field_inspector_name",
        )
        latitude = st.number_input("Latitude", value=0.0, format="%.6f")
        longitude = st.number_input("Longitude", value=0.0, format="%.6f")
        field_notes = st.text_area("Field Notes", "")
        photo_upload = st.file_uploader(
            "Photo Evidence",
            type=["jpg", "jpeg", "png"],
            key="field_photo",
        )

        if st.button("Upload Field Data", key="field_upload_btn"):
            try:
                data = {
                    "asset_id": field_asset_id,
                    "latitude": latitude,
                    "longitude": longitude,
                    "inspector_name": inspector_name,
                    "notes": field_notes,
                }
                request_kwargs: Dict[str, Any] = {"data": data, "timeout": 30}
                if photo_upload is not None:
                    request_kwargs["files"] = {
                        "photo": (
                            photo_upload.name,
                            photo_upload.getvalue(),
                            photo_upload.type or "application/octet-stream",
                        )
                    }
                resp = requests.post(
                    f"{API_URL}/v1/agents/asset_appraisal/upload_field_data",
                    **request_kwargs,
                )
                resp.raise_for_status()
                st.success("Field data uploaded.")
                st.json(resp.json())
            except Exception as exc:
                st.error(f"Failed to upload field data: {exc}")

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
            st.session_state["last_merged_df"] = pd.read_csv(uploaded_review)
            st.success("Loaded review dataset from uploaded CSV.")
        except Exception as e:
            st.error(f"Could not read uploaded CSV: {e}")

    if "last_merged_df" not in st.session_state:
        st.info("Run the agent (previous tab) or upload an AI outputs CSV to load results for review.")
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

        st.markdown("#### 2) Compute agreement score")
        if st.button("Compute agreement score"):
            if "ai_decision" in edited.columns and "human_decision" in edited.columns:
                agree = (edited["ai_decision"] == edited["human_decision"]).astype(int)
                score = float(agree.mean()) if len(agree) else 0.0
                st.success(f"Agreement score (AI vs human): {score:.3f}")
                st.session_state["last_agreement_score"] = score
            else:
                st.warning("Missing decision columns to compute score.")

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

    st.markdown("---")
    st.markdown("#### Production Model")
    try:
        resp = requests.get(f"{API_URL}/v1/training/production_meta", timeout=5)
        if resp.ok:
            st.json(resp.json())
        else:
            st.info("No production model yet.")
    except Exception as e:
        st.warning(f"Could not load production meta: {e}")
