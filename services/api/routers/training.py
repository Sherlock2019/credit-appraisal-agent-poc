# services/api/routers/training.py
from __future__ import annotations

import os
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from services.api.agents import list_registered_tasks
from services.api.agents.trainer import train_agent as hf_train_agent

# Model utils from the credit appraisal agent
from agents.credit_appraisal import model_utils as MU

router = APIRouter(prefix="/v1/training", tags=["training"])


# ───────────────────────────────────────────────────────────────
# Pydantic models
# ───────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    feedback_csvs: List[str] = Field(..., description="Absolute paths to feedback CSVs")
    user_name: str
    agent_name: str = "credit_appraisal"
    algo_name: str = "credit_lr"

    @field_validator("feedback_csvs")
    @classmethod
    def _check_csvs_exist(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("feedback_csvs must be a non-empty list of paths.")
        missing = [p for p in v if not os.path.exists(p)]
        if missing:
            raise ValueError(f"The following feedback_csvs do not exist: {missing}")
        return v


class PromoteRequest(BaseModel):
    # Optional, if omitted we will promote the latest trained model
    model_name: Optional[str] = None


class HFTrainRequest(BaseModel):
    task_name: str = Field(..., description="Registered Hugging Face task identifier")
    dataset_path: str = Field(..., description="Path to a Kaggle dataset (CSV/TSV)")
    text_col: str = Field(..., description="Column containing free-text features")
    label_col: str = Field(..., description="Target column")
    num_train_epochs: int = Field(2, ge=1)
    learning_rate: float = Field(2e-5, gt=0)
    per_device_train_batch_size: int = Field(8, ge=1)
    weight_decay: float = Field(0.01, ge=0)
    warmup_steps: int = Field(0, ge=0)
    max_train_samples: Optional[int] = Field(None, ge=1)
    evaluation_split: float = Field(0.2, ge=0.0, lt=1.0)
    seed: int = Field(42)

    @field_validator("dataset_path")
    @classmethod
    def _dataset_exists(cls, v: str) -> str:
        if not os.path.exists(v):
            raise ValueError(f"Dataset not found: {v}")
        return v

    @field_validator("task_name")
    @classmethod
    def _task_registered(cls, v: str) -> str:
        if v not in list_registered_tasks():
            raise ValueError(f"Unknown task '{v}'. Available: {sorted(list_registered_tasks())}")
        return v


# ───────────────────────────────────────────────────────────────
# Endpoints
# ───────────────────────────────────────────────────────────────

@router.post("/train")
def train(req: TrainRequest) -> Dict[str, Any]:
    """
    Train a candidate model using one or more human-feedback CSVs.
    Payload contract:
      {
        "feedback_csvs": ["/abs/path/feedback1.csv", "..."],
        "user_name": "Alice",
        "agent_name": "credit_appraisal",
        "algo_name": "credit_lr"
      }
    """
    try:
        resp = MU.fit_candidate_on_feedback(
            feedback_csvs=req.feedback_csvs,
            user_name=req.user_name,
            agent_name=req.agent_name,
            algo_name=req.algo_name,
        )
        return resp
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Training failed: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Training failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")


@router.post("/promote")
def promote(req: Optional[PromoteRequest] = None) -> Dict[str, Any]:
    """
    Promote a trained model to production.

    - If body is missing or model_name is None/blank: promote the latest trained model.
    - If model_name is given (either a filename or a full path), we sanitize to basename.
    """
    try:
        # Fallback when body is missing entirely
        model_name = None if req is None else (req.model_name or None)

        if not model_name:
            # auto-promote the latest trained model
            return MU.promote_last_trained_to_production()

        # allow callers sending full absolute path — sanitize to filename
        model_name = os.path.basename(model_name)
        return MU.promote_to_production(model_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Promote failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Promote failed: {e}")


@router.get("/production_meta")
def production_meta() -> Dict[str, Any]:
    """
    Returns:
      { "has_production": bool, "meta": {...} }
    """
    try:
        return MU.get_production_meta()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch production meta: {e}")


@router.get("/list_models")
def list_models(kind: str = "trained") -> Dict[str, Any]:
    """
    List available models.
      kind ∈ {"trained","production"}
    """
    try:
        models = MU.list_available_models(kind=kind)
        return {"kind": kind, "models": models}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {e}")


@router.get("/hf/tasks")
def list_hf_tasks() -> Dict[str, Any]:
    """Expose the available Hugging Face task registrations."""

    return {"tasks": list_registered_tasks()}


@router.post("/hf/train")
def hf_train(req: HFTrainRequest) -> Dict[str, Any]:
    """Kick off a lightweight fine-tuning job for a Hugging Face model."""

    try:
        output_dir = hf_train_agent(
            task_name=req.task_name,
            dataset_path=req.dataset_path,
            text_col=req.text_col,
            label_col=req.label_col,
            num_train_epochs=req.num_train_epochs,
            learning_rate=req.learning_rate,
            per_device_train_batch_size=req.per_device_train_batch_size,
            weight_decay=req.weight_decay,
            warmup_steps=req.warmup_steps,
            max_train_samples=req.max_train_samples,
            evaluation_split=req.evaluation_split,
            seed=req.seed,
        )
        return {
            "status": "ok",
            "output_dir": str(output_dir),
            "task": req.task_name,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HF training failed: {e}")
