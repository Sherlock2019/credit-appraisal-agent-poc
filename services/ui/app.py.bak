# services/ui/app.py
import os
import io
import re
import datetime
import pandas as pd
import numpy as np
import requests
import streamlit as st
import altair as alt

# ─────────────────────────────────────────────
# CONFIG
API_URL = os.getenv("API_URL", "http://localhost:8090")
RUNS_DIR = os.path.expanduser("~/demo-library/services/api/.runs")
os.makedirs(RUNS_DIR, exist_ok=True)

st.set_page_config(page_title="AI Credit Appraisal Platform", layout="wide")

# ─────────────────────────────────────────────
# HEADER — USER INFO + SECURITY
st.title("💳 AI Credit Appraisal Platform")
st.caption("Generate, sanitize, and appraise credit datasets securely — with tunable lending policies.")

with st.container():
    st.markdown("### 🧑 User Identity & Security Info")
    col1, col2, col3 = st.columns([1.5, 1.5, 1])
    with col1:
        user_name = st.text_input("Your Name (required)", value="", placeholder="e.g. Alice Nguyen")
    with col2:
        user_email = st.text_input("Email (required)", value="", placeholder="e.g. alice@bank.com")
    with col3:
        flag_session = st.checkbox("⚠️ Flag for Security Review", value=False)

    if not user_name or not user_email:
        st.warning("Please enter your name and email before proceeding.")
        st.stop()

st.session_state["user_info"] = {
    "name": user_name.strip(),
    "email": user_email.strip(),
    "flagged": flag_session,
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

# ─────────────────────────────────────────────
# TABS
tab_gen, tab_clean, tab_run, tab_train = st.tabs([
    "🏦 Synthetic Data Generator",
    "🧹 Anonymize & Sanitize Data",
    "📤 Run & Compare (with Decision Metrics)",
    "🔁 Training (Feedback → Retrain)"
])

# ─────────────────────────────────────────────
# UTILITIES

BANNED_NAMES = {"race", "gender", "religion", "ethnicity", "ssn", "national_id"}
PII_COLS = {"customer_name", "name", "email", "phone", "address", "ssn", "national_id", "dob"}

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d\-\s]{6,}\d")

def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.duplicated(keep="last")]

def scrub_text_pii(s):
    """Remove emails/phones within free-text cells."""
    if not isinstance(s, str):
        return s
    s = EMAIL_RE.sub("", s)
    s = PHONE_RE.sub("", s)
    return s.strip()

def drop_pii_columns(df: pd.DataFrame):
    """Drop PII columns and scrub any residual PII-like text in remaining object cols."""
    original_cols = list(df.columns)
    keep_cols = [c for c in original_cols if all(k not in c.lower() for k in PII_COLS)]
    dropped = [c for c in original_cols if c not in keep_cols]
    out = df[keep_cols].copy()
    for c in out.select_dtypes(include="object"):
        out[c] = out[c].apply(scrub_text_pii)
    return dedupe_columns(out), dropped

def strip_policy_banned(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that violate API policy (by name)."""
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

# 🔧 schema harmonization for API
def to_agent_schema(df: pd.DataFrame) -> pd.DataFrame:
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
    if "current_loans" not in out.columns:
        out["current_loans"] = rng.integers(0, 5, n)
    if "requested_amount" not in out.columns:
        out["requested_amount"] = out.get("loan_amount", 0)
    if "loan_term_months" not in out.columns:
        out["loan_term_months"] = out.get("loan_duration_months", 0)
    return dedupe_columns(out)

# ─────────────────────────────────────────────
# DATA GENERATORS
def generate_raw_synthetic(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    names = ["Alice Nguyen","Bao Tran","Chris Do","Duy Le","Emma Tran",
             "Felix Nguyen","Giang Ho","Hanh Vo","Ivan Pham","Julia Ngo"]
    emails = [f"{n.split()[0].lower()}.{n.split()[1].lower()}@gmail.com" for n in names]
    addrs = [
        "23 Elm St, Boston, MA","19 Pine Ave, San Jose, CA","14 High St, London, UK",
        "55 Nguyen Hue, Ho Chi Minh","78 Oak St, Chicago, IL","10 Broadway, New York, NY",
        "8 Rue Lafayette, Paris, FR","21 Königstr, Berlin, DE","44 Maple Dr, Los Angeles, CA","22 Bay St, Toronto, CA"
    ]
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
        "collateral_type": rng.choice(["house","car","land","deposit"], n),
        "co_loaners": rng.choice([0,1,2], n, p=[0.7,0.25,0.05]),
        "credit_score": rng.integers(300, 850, n),
        "existing_debt": rng.integers(0, 50_000, n),
        "assets_owned": rng.integers(10_000, 300_000, n),
    })
    eps = 1e-9
    df["DTI"] = df["existing_debt"] / (df["income"] + eps)
    df["LTV"] = df["loan_amount"] / (df["collateral_value"] + eps)
    df["CCR"] = df["collateral_value"] / (df["loan_amount"] + eps)
    df["ITI"] = (df["loan_amount"] / (df["loan_duration_months"] + eps)) / (df["income"] + eps)
    df["CWI"] = ((1 - df["DTI"]).clip(0, 1)) * ((1 - df["LTV"]).clip(0, 1)) * (df["CCR"].clip(0, 3))
    return dedupe_columns(df)

def generate_anon_synthetic(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "application_id": [f"APP_{i:04d}" for i in range(1, n + 1)],
        "age": rng.integers(21, 65, n),
        "income": rng.integers(25_000, 150_000, n),
        "employment_length": rng.integers(0, 30, n),
        "loan_amount": rng.integers(5_000, 100_000, n),
        "loan_duration_months": rng.choice([12, 24, 36, 48, 60, 72], n),
        "collateral_value": rng.integers(8_000, 200_000, n),
        "collateral_type": rng.choice(["house","car","land","deposit"], n),
        "co_loaners": rng.choice([0,1,2], n, p=[0.7,0.25,0.05]),
        "credit_score": rng.integers(300, 850, n),
        "existing_debt": rng.integers(0, 50_000, n),
        "assets_owned": rng.integers(10_000, 300_000, n),
    })
    eps = 1e-9
    df["DTI"] = df["existing_debt"] / (df["income"] + eps)
    df["LTV"] = df["loan_amount"] / (df["collateral_value"] + eps)
    df["CCR"] = df["collateral_value"] / (df["loan_amount"] + eps)
    df["ITI"] = (df["loan_amount"] / (df["loan_duration_months"] + eps)) / (df["income"] + eps)
    df["CWI"] = ((1 - df["DTI"]).clip(0, 1)) * ((1 - df["LTV"]).clip(0, 1)) * (df["CCR"].clip(0, 3))
    return dedupe_columns(df)

# ─────────────────────────────────────────────
# 🏦 TAB 1 — Synthetic Data Generator
with tab_gen:
    st.subheader("🏦 Synthetic Credit Data Generator")

    rows = st.slider("Number of rows to generate", 50, 1000, 200, step=50)
    colA, colB = st.columns(2)

    with colA:
        if st.button("🔴 Generate RAW Synthetic Data (with PII)", use_container_width=True):
            raw_df = append_user_info(generate_raw_synthetic(rows))
            st.session_state.synthetic_raw_df = raw_df
            raw_path = save_to_runs(raw_df, "synthetic_raw")
            st.success(f"Generated RAW (PII) dataset with {rows} rows. Saved to {raw_path}")
            st.dataframe(raw_df.head(10), use_container_width=True)
            st.download_button("⬇️ Download RAW CSV",
                               raw_df.to_csv(index=False).encode("utf-8"),
                               os.path.basename(raw_path), "text/csv")

    with colB:
        if st.button("🟢 Generate ANON Synthetic Data (ready for agent)", use_container_width=True):
            anon_df = append_user_info(generate_anon_synthetic(rows))
            st.session_state.synthetic_df = anon_df
            anon_path = save_to_runs(anon_df, "synthetic_anon")
            st.success(f"Generated ANON dataset with {rows} rows. Saved to {anon_path}")
            st.dataframe(anon_df.head(10), use_container_width=True)
            st.download_button("⬇️ Download ANON CSV",
                               anon_df.to_csv(index=False).encode("utf-8"),
                               os.path.basename(anon_path), "text/csv")

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
        st.download_button("⬇️ Download Clean Data",
                           sanitized.to_csv(index=False).encode("utf-8"),
                           os.path.basename(fpath),
                           "text/csv")
    else:
        st.info("Choose a CSV to see the sanitize flow.", icon="ℹ️")

# ─────────────────────────────────────────────
# 📤 TAB 3 — Run & Compare (Decision Metrics)
with tab_run:
    st.subheader("📤 Run Credit Appraisal Agent")

    # === Decision Metrics (Guardrails & Thresholds) ===
    with st.expander("⚙️ Decision Metrics (Guardrails & Thresholds)", expanded=True):
        st.markdown("Use realistic bank defaults — tweak to see how approvals change.")
        c1, c2, c3 = st.columns(3)
        with c1:
            max_dti = st.slider(
                "Max Debt-to-Income (DTI)", 0.0, 1.0, 0.45, 0.01,
                help="Total debt / income. Many lenders cap around 40–45%."
            )
            min_emp_years = st.number_input(
                "Min Employment Years", 0, 40, 2,
                help="Minimum continuous employment; 2+ years is common."
            )
            min_credit_hist = st.number_input(
                "Min Credit History (years)", 0, 40, 3,
                help="Minimum length of borrower credit history."
            )
        with c2:
            salary_floor = st.number_input(
                "Minimum Monthly Salary ($)", 500, 20000, 3000, step=500,
                help="Basic affordability floor."
            )
            max_delinquencies = st.number_input(
                "Max Delinquencies", 0, 10, 2,
                help="Max allowed past-due accounts."
            )
            max_current_loans = st.number_input(
                "Max Current Loans", 0, 10, 3,
                help="Cap on simultaneous active loans."
            )
        with c3:
            req_min = st.number_input(
                "Requested Amount Min ($)", 0, 1_000_000, 1_000, step=1000
            )
            req_max = st.number_input(
                "Requested Amount Max ($)", 0, 1_000_000, 200_000, step=1000
            )
            loan_terms = st.multiselect(
                "Allowed Loan Terms (months)", [12, 24, 36, 48, 60, 72],
                default=[12, 24, 36, 48, 60]
            )

        st.markdown("#### 🧮 Debt Pressure Controls")
        d1, d2, d3 = st.columns(3)
        with d1:
            min_income_debt_ratio = st.slider(
                "Min Income / (Compounded Debt) Ratio", 0.10, 2.00, 0.35, 0.01,
                help="Monthly income ÷ (existing monthly debt + factor × requested monthly). Higher is safer."
            )
        with d2:
            compounded_debt_factor = st.slider(
                "Compounded Debt Factor (× requested)", 0.5, 3.0, 1.0, 0.1,
                help="How much of the requested amount counts into 'compounded debt'."
            )
        with d3:
            monthly_debt_relief = st.slider(
                "Monthly Debt Relief Factor", 0.10, 1.00, 0.50, 0.05,
                help="Simulated relief on payment burden (e.g., 0.5 ≈ 50% relief)."
            )

        st.markdown("---")
        use_target = st.toggle("🎯 Use target approval rate (auto-quantile threshold)", value=False)
        random_band = st.toggle("🎲 Randomize approval band (20–60%) when no target is set", value=True)

        if use_target:
            target_rate = st.slider("Target approval rate", 0.05, 0.95, 0.40, 0.01)
            threshold = None
        else:
            threshold = st.slider("Model score threshold", 0.0, 1.0, 0.45, 0.01)
            target_rate = None

    # === Data Source ===
    data_choice = st.selectbox(
        "Select Data Source",
        [
            "Use sample dataset",
            "Use synthetic (ANON)",
            "Use synthetic (RAW – auto-sanitize)",
            "Use anonymized dataset",
            "Upload manually"
        ]
    )
    use_llm = st.checkbox("Use LLM narrative", value=False)
    agent_name = "credit_appraisal"

    # Manual upload for this tab
    if data_choice == "Upload manually":
        up = st.file_uploader("Upload your CSV", type=["csv"], key="manual_upload_run")
        if up is not None:
            st.session_state.manual_upload_run = {"name": up.name, "bytes": up.getvalue()}
            st.success(f"File staged: {up.name} ({len(st.session_state.manual_upload_run['bytes'])} bytes)")

    if st.button("🚀 Run Agent", use_container_width=True):
        try:
            files = None
            data = {
                "use_sample": "false",
                "use_llm_narrative": str(use_llm).lower(),
                # business filters
                "min_employment_years": str(min_emp_years),
                "max_debt_to_income": str(max_dti),
                "min_credit_history_length": str(min_credit_hist),
                "max_num_delinquencies": str(max_delinquencies),
                "max_current_loans": str(max_current_loans),
                "requested_amount_min": str(req_min),
                "requested_amount_max": str(req_max),
                "loan_term_months_allowed": ",".join(map(str, loan_terms)) if loan_terms else "",
                # debt pressure controls
                "min_income_debt_ratio": str(min_income_debt_ratio),
                "compounded_debt_factor": str(compounded_debt_factor),
                "monthly_debt_relief": str(monthly_debt_relief),
                "salary_floor": str(salary_floor),
                # model controls
                "threshold": "" if threshold is None else str(threshold),
                "target_approval_rate": "" if target_rate is None else str(target_rate),
                # support both names for the API
                "random_band": str(random_band).lower(),
                "random_approval_band": str(random_band).lower(),
            }

            def prep_and_pack(df: pd.DataFrame, filename: str):
                safe = dedupe_columns(df)
                safe, _ = drop_pii_columns(safe)
                safe = strip_policy_banned(safe)
                safe = to_agent_schema(safe)
                buf = io.StringIO()
                safe.to_csv(buf, index=False)
                return {"file": (filename, buf.getvalue().encode("utf-8"), "text/csv")}

            if data_choice == "Use sample dataset":
                data["use_sample"] = "true"

            elif data_choice == "Use synthetic (ANON)":
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

            elif data_choice == "Upload manually":
                up_bytes = st.session_state.get("manual_upload_run")
                if up_bytes is None:
                    st.warning("Please upload a CSV first.")
                    st.stop()
                try:
                    tmp_df = pd.read_csv(io.BytesIO(up_bytes["bytes"]))
                    files = prep_and_pack(tmp_df, up_bytes["name"])
                except Exception:
                    files = {"file": (up_bytes["name"], up_bytes["bytes"], "text/csv")}

            r = requests.post(f"{API_URL}/v1/agents/{agent_name}/run", data=data, files=files, timeout=180)
            if r.status_code != 200:
                st.error(f"Run failed ({r.status_code}): {r.text}")
                st.stop()

            res = r.json()
            st.session_state.last_run_id = res.get("run_id")
            result = res.get("result", {})
            summary = result.get("summary", {}) or {}

            st.success(f"✅ Run succeeded! Run ID: {st.session_state.last_run_id}")

            # Summary metrics
            total = int(summary.get("count", 0))
            approved = int(summary.get("approved", 0))
            denied = int(summary.get("denied", 0))
            used_thr = summary.get("threshold_used")
            used_rate = summary.get("target_rate_used")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total", total)
            c2.metric("Approved", approved)
            c3.metric("Denied", denied)
            if used_thr is not None:
                c4.metric("Threshold used", f"{used_thr:.3f}")
            elif used_rate is not None:
                c4.metric("Target approval used", f"{used_rate:.2%}")
            else:
                c4.metric("Threshold/Target", "—")

            # Acceptance chart
            chart_df = pd.DataFrame({"Outcome": ["Approved", "Denied"], "Count": [approved, denied]})
            chart = alt.Chart(chart_df).mark_bar().encode(
                x=alt.X("Outcome:N", title=None),
                y=alt.Y("Count:Q"),
                color="Outcome:N",
                tooltip=["Outcome", "Count"]
            ).properties(height=220)
            st.altair_chart(chart, use_container_width=True)

            # Tables
            scores = result.get("scores", [])
            expl = result.get("explanations", [])
            if scores:
                st.markdown("### Scores (sample)")
                st.dataframe(pd.DataFrame(scores).head(25), use_container_width=True)
            if expl:
                st.markdown("### Explanations (sample)")
                df_expl = pd.DataFrame(expl)
                sel_cols = [c for c in [
                    "application_id","decision","top_feature","explanation",
                    "proposed_loan_option","proposed_consolidation_loan"
                ] if c in df_expl.columns]
                st.dataframe(df_expl[sel_cols].head(25), use_container_width=True)

            # Download links
            rid = st.session_state.last_run_id
            cdl1, cdl2, cdl3, cdl4, cdl5 = st.columns(5)
            with cdl1: st.markdown(f"[⬇️ PDF report]({API_URL}/v1/runs/{rid}/report?format=pdf)")
            with cdl2: st.markdown(f"[⬇️ Scores CSV]({API_URL}/v1/runs/{rid}/report?format=scores_csv)")
            with cdl3: st.markdown(f"[⬇️ Explanations CSV]({API_URL}/v1/runs/{rid}/report?format=explanations_csv)")
            with cdl4: st.markdown(f"[⬇️ Merged CSV]({API_URL}/v1/runs/{rid}/report?format=csv)")
            with cdl5: st.markdown(f"[⬇️ JSON]({API_URL}/v1/runs/{rid}/report?format=json)")

        except Exception as e:
            st.exception(e)

    # Quick re-download
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
# 🔁 TAB 4 — Training (Human Feedback → Retrain)
with tab_train:
    st.subheader("🔁 Human Feedback → Retrain")

    # ——— Review queue ———
    st.markdown("#### Review Queue (recent results)")
    default_results = os.path.join(RUNS_DIR, "latest", "results.csv")
    results_path = st.text_input("Results CSV path", default_results)

    df_results = None
    try:
        df_results = pd.read_csv(os.path.expanduser(results_path))
        colf1, colf2 = st.columns(2)
        with colf1:
            score_min, score_max = st.slider("Score range filter", 0.0, 1.0, (0.0, 1.0), 0.01)
        with colf2:
            decision_filter = st.multiselect("Decision filter", ["approved","denied"], default=[])

        if "score" in df_results.columns:
            df_results = df_results[(df_results["score"] >= score_min) & (df_results["score"] <= score_max)]
        if decision_filter and "decision" in df_results.columns:
            df_results = df_results[df_results["decision"].isin(decision_filter)]

        st.dataframe(dedupe_columns(df_results).head(200), use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load results: {e}")

    st.markdown("---")

    # ——— Feedback builder ———
    st.markdown("#### Create Feedback")
    c1, c2 = st.columns([1,1])
    with c1:
        application_id = st.text_input("application_id (required)", "")
        y_true = st.selectbox(
            "Ground-truth outcome (y_true)", [1, 0, -1], index=2,
            help="1=approved/good, 0=declined/bad, -1=unknown/pending"
        )
        label_confidence = st.slider("Label confidence", 0.0, 1.0, 0.9, 0.05)
        reason_codes = st.multiselect("Reason codes", ["DTI_HIGH","SHORT_HISTORY","DELINQ","FRAUD","POLICY_DENY"])
        notes = st.text_area("Notes")
        reviewer = st.session_state.get("user_info", {}).get("name") or "anonymous"

    with c2:
        st.caption("Corrected features (optional)")
        corrected = {}
        if st.checkbox("Override salary"):
            corrected["salary"] = st.number_input("salary", 0.0, step=100.0)
        if st.checkbox("Override employment_years"):
            corrected["employment_years"] = st.number_input("employment_years", 0.0, step=0.5)
        if st.checkbox("Override credit_hist_years"):
            corrected["credit_hist_years"] = st.number_input("credit_hist_years", 0.0, step=0.5)
        if st.checkbox("Override dti"):
            corrected["dti"] = st.number_input("dti", 0.0, step=0.01)
        if st.checkbox("Override curr_loans"):
            corrected["curr_loans"] = st.number_input("curr_loans", 0, step=1)

    cbtn1, cbtn2 = st.columns([1,3])
    with cbtn1:
        if st.button("➕ Append feedback", use_container_width=True):
            if not application_id:
                st.error("application_id is required.")
            else:
                payload = {
                    "application_id": application_id,
                    "y_true": y_true,
                    "label_confidence": label_confidence,
                    "reason_codes": reason_codes,
                    "notes": notes,
                    "corrected_features": corrected,
                    "reviewer_id": reviewer,
                }
                try:
                    r = requests.post(f"{API_URL}/v1/training/feedback", json=payload, timeout=10)
                    r.raise_for_status()
                    st.success(r.json())
                except Exception as e:
                    st.error(f"Feedback failed: {e}")

    st.markdown("---")

    # ——— Retrain controls ———
    st.markdown("#### Retrain Model")
    ctrain1, ctrain2 = st.columns([2,1])
    with ctrain1:
        cutoff_date = st.date_input(
            "Use data after (optional)",
            datetime.date(2024, 1, 1),
            help="Exclude very old rows to reduce concept drift."
        )
        base_globs = [os.path.expanduser(results_path)] if results_path else []
        st.code({"base_csv_globs": base_globs, "cutoff_date": cutoff_date.strftime("%Y-%m-%d")}, language="json")

    with ctrain2:
        if st.button("🚀 Train candidate model", use_container_width=True):
            cfg = {
                "base_csv_globs": base_globs or [os.path.join(RUNS_DIR, "latest", "results.csv")],
                "cutoff_date": cutoff_date.strftime("%Y-%m-%d"),
            }
            try:
                r = requests.post(f"{API_URL}/v1/training/train", json=cfg, timeout=10)
                r.raise_for_status()
                resp = r.json()
                st.session_state["last_train_job"] = resp.get("job_id")
                st.info(resp)
            except Exception as e:
                st.error(f"Train launch failed: {e}")

    # ——— Optional: poll job status ———
    job_id = st.session_state.get("last_train_job", "")
    if job_id:
        st.markdown("##### Training Job Status")
        colj1, colj2 = st.columns([1,3])
        with colj1:
            st.text_input("job_id", job_id, disabled=True)
        with colj2:
            if st.button("🔄 Refresh status", use_container_width=True):
                try:
                    rs = requests.get(f"{API_URL}/v1/training/train/{job_id}/status", timeout=10).json()
                    st.write(rs)
                    if rs.get("metrics"):
                        st.success("✅ Training complete")
                except Exception as e:
                    st.error(f"Status check failed: {e}")

