# services/api/routers/training.py
from __future__ import annotations

import os
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

# Model utils from the credit appraisal agent
from agents.credit_appraisal import model_utils as MU
from services.api.agents.model_registry import MODEL_MAP
from services.api.agents.trainer import TRAINABLE_TASKS, train_agent

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
    """Request payload for Hugging Face fine-tuning."""

    task_name: str = Field(..., description="Task identifier registered in MODEL_MAP")
    dataset_path: str = Field(..., description="Path to the Kaggle dataset CSV")
    text_col: str = Field(..., description="Name of the text column")
    label_col: str = Field(..., description="Name of the label column")

    @field_validator("task_name")
    @classmethod
    def _validate_task(cls, value: str) -> str:
        if value not in MODEL_MAP:
            raise ValueError(f"Unsupported task '{value}'. Available: {sorted(MODEL_MAP)}")
        if value not in TRAINABLE_TASKS:
            raise ValueError(
                f"Task '{value}' is not supported for Hugging Face fine-tuning. Choose from: {sorted(TRAINABLE_TASKS)}"
            )
        return value

    @field_validator("dataset_path")
    @classmethod
    def _validate_dataset(cls, value: str) -> str:
        if not os.path.exists(value):
            raise ValueError(f"Dataset not found: {value}")
        return value


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


@router.post("/hf/train")
def train_huggingface(req: HFTrainRequest) -> Dict[str, Any]:
    """Fine-tune a registered Hugging Face model using a Kaggle dataset."""

    try:
        save_path = train_agent(
            task_name=req.task_name,
            dataset_path=req.dataset_path,
            text_col=req.text_col,
            label_col=req.label_col,
        )
        return {"status": "ok", "model_path": save_path}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")


@router.get("/hf/tasks")
def list_huggingface_tasks() -> Dict[str, Any]:
    """Expose the task-to-model mapping used by the agent library."""

    return {
        "tasks": MODEL_MAP,
        "trainable_tasks": sorted(TRAINABLE_TASKS),
    }
