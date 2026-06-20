"""Trace models for agent pipeline observability."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from src.schemas import AgentOutput

StepPhase = Literal[
    "ingest",
    "prompt",
    "llm",
    "parse",
    "validate",
    "retry",
    "evaluate",
    "threshold",
    "complete",
    "error",
]
StepStatus = Literal["info", "success", "warning", "error"]


class TraceStep(BaseModel):
    step_id: str
    phase: StepPhase
    title: str
    status: StepStatus = "info"
    elapsed_ms: int = 0
    message: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class RecordTrace(BaseModel):
    task_id: str
    input_record: dict[str, Any]
    steps: list[TraceStep] = Field(default_factory=list)
    output: AgentOutput | None = None
    error: str | None = None
    latency_ms: int = 0


class RunSummary(BaseModel):
    total_records: int
    sent: int
    suppressed: int
    channel_distribution: dict[str, int]
    average_personalization_score: float
    max_safety_violations: int
    average_latency_ms: float
    threshold_pass_rate: float


class RunTrace(BaseModel):
    run_id: str
    mock: bool
    provider: str
    model: str
    started_at: str
    total_latency_ms: int
    summary: RunSummary
    records: list[RecordTrace] = Field(default_factory=list)
