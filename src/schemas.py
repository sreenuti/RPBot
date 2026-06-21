"""Pydantic schemas for input records and agent output."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Channel = Literal["sms", "email", "push", "none"]


class FlexibleModel(BaseModel):
    """Base model that accepts unknown fields from hold-out datasets."""

    model_config = ConfigDict(extra="allow")


class Profile(FlexibleModel):
    first_name: str | None = None


class InputContext(FlexibleModel):
    property_name: str | None = None
    move_date_target: str | None = None
    last_interaction: str | None = None
    timezone: str = "UTC"
    language: str = "en"
    profile: Profile = Field(default_factory=Profile)


class Consent(FlexibleModel):
    email_opt_in: bool = False
    sms_opt_in: bool = False
    voice_opt_in: bool = False
    push_opt_in: bool = False
    global_opt_out: bool = False


class Constraints(FlexibleModel):
    no_pii_leak: bool = True
    no_sensitive_discrimination: bool = True
    include_opt_out_instructions: bool = False
    primary_cta: str | None = None
    quiet_hours: dict[str, Any] | None = None
    send_at: str | None = None
    next_action: dict[str, Any] | None = None


class Assertions(FlexibleModel):
    required_states: list[str] = Field(default_factory=list)
    constraints: Constraints = Field(default_factory=Constraints)


class Thresholds(FlexibleModel):
    p95_latency_ms: int | None = None
    personalization_score_min: float | None = None
    reply_classification_f1_min: float | None = None
    safety_violations_max: int | None = None


class InputRecord(FlexibleModel):
    task_id: str
    persona: str | None = None
    lifecycle_stage: str | None = None
    consent: Consent = Field(default_factory=Consent)
    channel_preferences: list[str] = Field(default_factory=list)
    input: InputContext = Field(default_factory=InputContext)
    assertions: Assertions = Field(default_factory=Assertions)
    thresholds: Thresholds = Field(default_factory=Thresholds)
    expected: dict[str, Any] | None = None


class NextMessage(BaseModel):
    channel: Channel | None = None
    send_at: str | None = None
    subject: str | None = None
    body: str | None = None
    cta: dict[str, Any] | None = None


class NextAction(BaseModel):
    type: str
    details: dict[str, Any] | None = None


class QualityMetrics(BaseModel):
    personalization_score: float = 0.0
    safety_violations: int = 0
    latency_ms: int | None = None


class AgentOutput(BaseModel):
    task_id: str
    should_send: bool
    next_message: NextMessage
    next_action: NextAction
    reasoning: str
    quality: QualityMetrics = Field(default_factory=QualityMetrics)
