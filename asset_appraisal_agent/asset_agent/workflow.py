from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


_ASSET_TYPES = [
    "Residential Property",
    "Condominium",
    "Townhouse",
    "Commercial Property",
    "Vehicle",
    "Heavy Equipment",
    "Agricultural Land",
]

_STAGE_NAMES = [
    "kyc_screening",
    "document_validation",
    "valuation_modelling",
    "field_inspection",
    "credit_committee",
]


@dataclass
class AssetAppraisalResult:
    """Structured result produced for a single loan application."""

    application_id: str
    asset_type: str
    collateral_value: float
    collateral_status: str
    verification_stage: str
    confidence: float
    legitimacy_score: float
    include_in_credit: bool
    notes: str
    last_updated: str
    workflow_trace: List[Dict[str, Any]]
    loan_amount_declared: float | None = None
    borrower_segment: str | None = None

    def to_record(self) -> Dict[str, Any]:
        data = asdict(self)
        data["workflow_trace"] = json.dumps(self.workflow_trace, ensure_ascii=False)
        return data


def generate_synthetic_loans(
    n_loans: int = 80,
    collateral_ratio: float = 0.8,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Generate a synthetic set of loan requests with collateral hints."""

    rng = np.random.default_rng(seed)
    loan_ids = [f"APP-{1000 + i}" for i in range(n_loans)]
    loan_amounts = rng.integers(20_000, 350_000, size=n_loans)
    income = rng.integers(30_000, 180_000, size=n_loans)
    collateral_flags = rng.uniform(0, 1, size=n_loans) < collateral_ratio

    rows = []
    for idx, loan_id in enumerate(loan_ids):
        has_collateral = bool(collateral_flags[idx])
        base_amount = float(loan_amounts[idx])
        asset_type = random.choice(_ASSET_TYPES)
        declared_value = base_amount * random.uniform(0.9, 1.6) if has_collateral else 0.0
        rows.append(
            {
                "application_id": loan_id,
                "loan_amount": base_amount,
                "income": float(income[idx]),
                "customer_segment": random.choice(["Retail", "SME", "Corporate"]),
                "has_collateral": has_collateral,
                "declared_collateral_value": round(declared_value, 2),
                "asset_type_hint": asset_type if has_collateral else None,
            }
        )
    return pd.DataFrame(rows)


class AssetAppraisalWorkflow:
    """Orchestrates the collateral verification workflow for a batch of loans."""

    def __init__(self, random_seed: int | None = 42) -> None:
        self.random = random.Random(random_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, loans: pd.DataFrame | Sequence[Dict[str, Any]]) -> pd.DataFrame:
        """Evaluate incoming loans and return collateral statuses as DataFrame."""

        if isinstance(loans, pd.DataFrame):
            df = loans.copy()
        else:
            df = pd.DataFrame(list(loans))
        if df.empty:
            raise ValueError("No loan records supplied to asset appraisal workflow.")

        results = [self._evaluate_row(row) for row in df.to_dict(orient="records")]
        return pd.DataFrame([result.to_record() for result in results])

    def generate_synthetic(self, n_loans: int = 80, collateral_ratio: float = 0.8) -> pd.DataFrame:
        loans = generate_synthetic_loans(n_loans=n_loans, collateral_ratio=collateral_ratio, seed=self.random.randint(0, 9999))
        return self.run(loans)

    def export(self, df: pd.DataFrame, destination: str | Path) -> Path:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return path

    # ------------------------------------------------------------------
    # Internal logic
    # ------------------------------------------------------------------
    def _evaluate_row(self, row: Dict[str, Any]) -> AssetAppraisalResult:
        app_id = self._resolve_application_id(row)
        loan_amount = float(self._safe_float(row.get("loan_amount") or row.get("requested_amount"), default=0.0))
        declared_value = float(self._safe_float(row.get("declared_collateral_value"), default=loan_amount * self.random.uniform(0.85, 1.5)))
        has_collateral = self._resolve_has_collateral(row)
        asset_type = self._resolve_asset_type(row, has_collateral)

        workflow_trace: List[Dict[str, Any]] = []
        status = "validated"
        verification_stage = "completed"
        include_in_credit = True
        notes: List[str] = []

        if not has_collateral:
            status = "missing"
            verification_stage = "no_collateral"
            include_in_credit = False
            declared_value = 0.0
            notes.append("No collateral declared by borrower.")
        else:
            stage_outcome = self._stage_runner(row, declared_value)
            workflow_trace = stage_outcome["trace"]
            status = stage_outcome["status"]
            verification_stage = stage_outcome["stage"]
            include_in_credit = stage_outcome["include"]
            declared_value = stage_outcome["collateral_value"]
            notes.extend(stage_outcome.get("notes", []))

        confidence = round(self.random.uniform(0.78, 0.97), 2)
        legitimacy = round(self.random.uniform(0.8, 0.99), 2)
        if status == "denied_fraud":
            confidence = round(max(0.1, confidence - 0.5), 2)
            legitimacy = round(min(0.4, legitimacy - 0.5), 2)
        elif status in {"pending_asset_review", "under_verification", "re_evaluate"}:
            confidence = round(max(0.4, confidence - 0.2), 2)
            legitimacy = round(max(0.5, legitimacy - 0.25), 2)

        return AssetAppraisalResult(
            application_id=app_id,
            asset_type=asset_type,
            collateral_value=round(declared_value, 2),
            collateral_status=status,
            verification_stage=verification_stage,
            confidence=confidence,
            legitimacy_score=legitimacy,
            include_in_credit=include_in_credit,
            notes="; ".join(notes) if notes else "",
            last_updated=datetime.utcnow().isoformat(),
            workflow_trace=workflow_trace,
            loan_amount_declared=round(loan_amount, 2) if loan_amount else None,
            borrower_segment=row.get("customer_segment"),
        )

    def _stage_runner(self, row: Dict[str, Any], declared_value: float) -> Dict[str, Any]:
        trace: List[Dict[str, Any]] = []
        status = "validated"
        stage = "completed"
        include_in_credit = True
        notes: List[str] = []
        collateral_value = declared_value * self.random.uniform(0.92, 1.18)

        for stage_name in _STAGE_NAMES:
            outcome = self._evaluate_stage(stage_name, row, collateral_value)
            trace.append({"stage": stage_name, **outcome})
            if outcome.get("decision") == "fail":
                if stage_name == "kyc_screening":
                    status = "denied_fraud"
                    stage = stage_name
                    include_in_credit = False
                    notes.append("KYC screening flagged suspected fraud.")
                    break
                elif stage_name in {"document_validation", "field_inspection"}:
                    status = "under_verification" if stage_name == "document_validation" else "re_evaluate"
                    stage = stage_name
                    include_in_credit = False
                    notes.append(outcome.get("note", "Manual verification required."))
                    break
                else:
                    status = "pending_asset_review"
                    stage = stage_name
                    include_in_credit = False
                    notes.append(outcome.get("note", "Awaiting manual approval."))
                    break
            elif outcome.get("decision") == "monitor":
                status = "monitor"
                stage = stage_name
                include_in_credit = True
                notes.append(outcome.get("note", "Collateral flagged for monitoring."))

            collateral_value = outcome.get("collateral_value", collateral_value)

        return {
            "status": status,
            "stage": stage,
            "include": include_in_credit,
            "collateral_value": collateral_value,
            "trace": trace,
            "notes": notes,
        }

    def _evaluate_stage(self, stage_name: str, row: Dict[str, Any], collateral_value: float) -> Dict[str, Any]:
        decision = "pass"
        note = ""
        stage_multiplier = 1.0

        # KYC — low probability of direct fraud flag
        if stage_name == "kyc_screening":
            if self.random.random() < 0.04:
                decision = "fail"
                note = "Identity or sanction list hit."
        elif stage_name == "document_validation":
            if self.random.random() < 0.12:
                decision = "fail"
                note = "Documents inconsistent with registry data."
            elif self.random.random() < 0.08:
                decision = "monitor"
                note = "Documents valid but require periodic refresh."
        elif stage_name == "valuation_modelling":
            loan_amount = self._safe_float(row.get("loan_amount")) or 0.0
            if loan_amount and collateral_value:
                ltv = loan_amount / collateral_value
                if ltv > 1.2:
                    stage_multiplier = 1.1
                    note = "Loan amount > 120% of collateral — flagged for inspection."
                    decision = "monitor"
            collateral_value *= self.random.uniform(0.95, 1.05)
        elif stage_name == "field_inspection":
            if self.random.random() < 0.1:
                decision = "fail"
                note = "Physical inspection found discrepancies."
                collateral_value *= self.random.uniform(0.7, 0.95)
        elif stage_name == "credit_committee":
            if self.random.random() < 0.05:
                decision = "fail"
                note = "Committee requested manual review."
            collateral_value *= self.random.uniform(0.97, 1.03)

        return {
            "decision": decision,
            "note": note,
            "collateral_value": collateral_value * stage_multiplier,
        }

    @staticmethod
    def _resolve_application_id(row: Dict[str, Any]) -> str:
        for key in ("application_id", "loan_id", "id", "app_id"):
            value = row.get(key)
            if value:
                return str(value)
        return f"APP-{random.randint(1000, 9999)}"

    @staticmethod
    def _safe_float(value: Any, default: float | None = None) -> float | None:
        if value in (None, ""):
            return default
        try:
            return float(value)
        except Exception:
            return default

    def _resolve_has_collateral(self, row: Dict[str, Any]) -> bool:
        flag = row.get("has_collateral")
        if flag is None:
            flag = row.get("collateral_flag")
        if flag is None:
            return self.random.random() > 0.1
        if isinstance(flag, str):
            return flag.strip().lower() in {"1", "true", "yes", "y"}
        return bool(flag)

    def _resolve_asset_type(self, row: Dict[str, Any], has_collateral: bool) -> str:
        if not has_collateral:
            return "None"
        for key in ("asset_type", "asset_category", "asset_type_hint"):
            value = row.get(key)
            if value:
                return str(value)
        return self.random.choice(_ASSET_TYPES)


def main() -> None:
    """CLI helper for manual testing."""

    import argparse

    parser = argparse.ArgumentParser(description="Run asset appraisal workflow")
    parser.add_argument("--loans", type=int, default=60, help="Number of synthetic loans to generate")
    parser.add_argument("--ratio", type=float, default=0.8, help="Share of loans that include collateral")
    parser.add_argument("--out", type=str, default="exports/sample_asset_appraisals.csv", help="Output CSV path")
    args = parser.parse_args()

    workflow = AssetAppraisalWorkflow()
    df = workflow.generate_synthetic(n_loans=args.loans, collateral_ratio=args.ratio)
    out_path = workflow.export(df, args.out)
    print(f"Saved synthetic asset appraisal export to {out_path}")


if __name__ == "__main__":
    main()
