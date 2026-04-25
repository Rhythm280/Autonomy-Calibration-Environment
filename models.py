"""
models.py — OpenEnv Pydantic contracts v2.0
Autonomy Calibration Environment
"""
from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field


class Action(BaseModel):
    type: str = Field(..., description="Action type from available_actions")
    payload: dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    task_id: str
    step: int
    state: dict[str, Any]
    history: list[dict[str, Any]] = Field(default_factory=list)
    available_actions: list[str]
    done: bool = False
    prompt: str = ""
    seed: Optional[int] = Field(default=None, description="Episode seed for reproducibility")
    
    # ─── Legacy Fields (Backward Compatibility for UI) ───────────────────
    context: str = ""
    task: str = ""
    action_to_evaluate: str = ""


class Reward(BaseModel):
    # OpenEnv v2 requirement: scores strictly in (0.01, 0.99)
    value: float = Field(..., ge=0.01, le=0.99)
    breakdown: dict[str, Any] = Field(default_factory=dict)
    raw: float = 0.0


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


class ResetRequest(BaseModel):
    task: str = Field(default="email_triage")
    seed: Optional[int] = Field(
        default=None,
        description="Integer seed for reproducibility. Same seed → same episode. None = auto-increment."
    )
