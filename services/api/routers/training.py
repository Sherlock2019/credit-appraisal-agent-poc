# services/api/routers/training.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
import os, pandas as pd, json, uuid, subprocess

router = APIRouter(prefix="/v1/training", tags=["training"])

ROOT = os.path.expanduser("~/demo-library")
RUNS_DIR = os.path.join(ROOT, "services", "api", ".runs")
FEEDBACK_DIR = os.path.join(RUNS_DIR, "feedback")
TRAIN_DIR = os.path.join(RUNS_DIR, "train")
os.makedirs(FEEDBACK_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)

class FeedbackItem(BaseModel):
    application_id: str
    y_true: int                 # 1 approved/good, 0 declined/bad, -1 unknown/pending
    label_confidence: float = 1.0
    reason_codes: list[str] = []
    notes: str = ""
    corrected_features: dict = {}  # e.g. {"salary":1200, "employment_years":5}
    # optional lineage (fill from your scoring output if available)
    y_pred: int | None = None
    p_pred: float | None = None
    model_version_at_score: str | None = None
    source_run_id: str | None = None
    dataset_hash_at_score: str | None = None
    reviewer_id: str = "anonymous"

@router.post("/feedback")
def submit_feedback(item: FeedbackItem):
    row = {
        "application_id": item.application_id,
        "timestamp_utc": datetime.utcnow().isoformat(),
        "reviewer_id": item.reviewer_id,
        "y_true": item.y_true,
        "label_confidence": item.label_confidence,
        "reason_codes": ",".join(item.reason_codes),
        "notes": item.notes,
        "y_pred": item.y_pred,
        "p_pred": item.p_pred,
        "model_version_at_score": item.model_version_at_score,
        "source_run_id": item.source_run_id,
        "dataset_hash_at_score": item.dataset_hash_at_score,
    }
    # flatten corrected feature edits with feature_ prefix
    for k, v in (item.corrected_features or {}).items():
        row[f"feature_{k}"] = v

    out = os.path.join(FEEDBACK_DIR, f"feedback_{datetime.utcnow():%Y%m}.csv")
    df = pd.DataFrame([row])
    df.to_csv(out, mode="a", header=not os.path.exists(out), index=False)
    return {"status": "ok", "file": out, "written": 1}

@router.post("/train")
def launch_training(config: dict):
    """
    Example config:
    {
      "base_csv_globs": ["~/demo-library/services/api/.runs/latest/results.csv"],
      "cutoff_date": "2024-01-01"
    }
    """
    job_id = str(uuid.uuid4())
    log_path = os.path.join(TRAIN_DIR, f"{job_id}.log")
    cfg_path = os.path.join(TRAIN_DIR, f"{job_id}.json")
    with open(cfg_path, "w") as f:
        json.dump(config, f)

    trainer = os.path.join(ROOT, "services", "train", "train_credit.py")
    if not os.path.exists(trainer):
        raise HTTPException(500, detail="trainer script not found")

    subprocess.Popen(
        ["python", trainer, "--job", job_id, "--config", cfg_path],
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT
    )
    return {"status": "queued", "job_id": job_id}

@router.get("/train/{job_id}/status")
def train_status(job_id: str):
    metrics_path = os.path.join(TRAIN_DIR, f"{job_id}.metrics.json")
    log_path = os.path.join(TRAIN_DIR, f"{job_id}.log")
    status = "running"
    metrics = None
    if os.path.exists(metrics_path):
        status = "complete"
        with open(metrics_path) as f:
            metrics = json.load(f)
    tail = ""
    if os.path.exists(log_path):
        with open(log_path) as f:
            tail = "".join(f.readlines()[-60:])
    return {"status": status, "metrics": metrics, "log_tail": tail}
