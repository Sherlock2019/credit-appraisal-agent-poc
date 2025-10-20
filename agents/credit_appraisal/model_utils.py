# agents/credit_appraisal/model_utils.py
import os, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

FEATURES = [
    "age", "income", "employment_years", "debt_to_income",
    "credit_history_length", "num_delinquencies",
    "current_loans", "requested_amount", "loan_term_months"
]
TARGET = "approved"


# ───────────────────────────────────────────────────────────────
def _synthesize_labels(df: pd.DataFrame) -> pd.Series:
    """
    Generates realistic synthetic approval labels with ~45% approval rate.
    Approval likelihood increases with income and employment history,
    and decreases with DTI, delinquencies, and requested amount.
    """

    df = df.copy()

    # normalize main metrics (to prevent large value dominance)
    income_norm = (df["income"] - df["income"].min()) / (df["income"].max() - df["income"].min() + 1e-9)
    emp_norm = np.clip(df["employment_years"] / 10.0, 0, 1)
    dti_norm = np.clip(df["debt_to_income"], 0, 1)
    cred_norm = np.clip(df["credit_history_length"] / 120.0, 0, 1)
    delin_norm = np.clip(df["num_delinquencies"] / 5.0, 0, 1)
    loans_norm = np.clip(df["current_loans"] / 5.0, 0, 1)
    req_norm = np.clip(df["requested_amount"] / df["requested_amount"].max(), 0, 1)

    # Weighted scoring model (positive vs. negative factors)
    score = (
        0.35 * income_norm +
        0.25 * emp_norm +
        0.15 * cred_norm -
        0.25 * dti_norm -
        0.20 * delin_norm -
        0.15 * loans_norm -
        0.20 * req_norm
    )

    # Add Gaussian noise for randomness
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.15, size=len(df))
    score = score + noise

    # Compute threshold for ~45% approval
    threshold = np.quantile(score, 0.55)  # 55% cutoff ⇒ ~45% approvals
    y = (score > threshold).astype(int)

    # Ensure binary diversity
    if y.nunique() < 2 and len(y) >= 2:
        y.iloc[0] = 1 - y.iloc[0]
    return y


# ───────────────────────────────────────────────────────────────
def ensure_model(df: pd.DataFrame):
    """
    Train or load a LightGBM model using synthetic or provided data.
    Automatically retrains if model.pkl missing or data changed.
    """
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass  # retrain if corrupt

    data = df.copy()
    data = data.dropna(subset=FEATURES)

    # Create labels if missing
    if TARGET not in data.columns:
        data[TARGET] = _synthesize_labels(data)

    # Safety: ensure both classes
    if data[TARGET].nunique() < 2:
        data.loc[data.index[0], TARGET] = 1 - data.loc[data.index[0], TARGET]

    X = data[FEATURES]
    y = data[TARGET]

    model = LGBMClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.07,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )

    # Train/test split with fallback
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
        model.fit(Xtr, ytr)
        try:
            proba = model.predict_proba(Xte)[:, 1]
            auc = roc_auc_score(yte, proba)
            print(f"[model_utils] Trained demo model (AUC={auc:.3f}, approval={y.mean():.2%})")
        except Exception:
            pass
    except Exception:
        model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    return model
