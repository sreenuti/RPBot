"""Quality evaluation and threshold comparison."""

from __future__ import annotations

from dataclasses import dataclass

from src.schemas import AgentOutput, InputRecord, QualityMetrics
from src.validator import count_safety_violations

PERSONALIZATION_WEIGHTS = {
    "first_name": 0.2,
    "property_name": 0.2,
    "move_date_target": 0.15,
    "amenity_interest": 0.15,
    "city_interest": 0.1,
    "cta": 0.2,
}


@dataclass
class ThresholdResult:
    passed: bool
    failures: list[str]


def _message_blob(output: AgentOutput) -> str:
    parts = [output.next_message.body or "", output.next_message.subject or ""]
    return " ".join(parts).lower()


def _available_context(record: InputRecord) -> dict[str, object]:
    profile_extra = getattr(record.input.profile, "model_extra", None) or {}
    available: dict[str, object] = {}
    if record.input.profile.first_name:
        available["first_name"] = record.input.profile.first_name
    if record.input.property_name:
        available["property_name"] = record.input.property_name
    if record.input.move_date_target:
        available["move_date_target"] = record.input.move_date_target
    if profile_extra.get("amenity_interest"):
        available["amenity_interest"] = profile_extra["amenity_interest"]
    if profile_extra.get("city_interest"):
        available["city_interest"] = profile_extra["city_interest"]
    if record.assertions.constraints.primary_cta:
        available["cta"] = record.assertions.constraints.primary_cta
    return available


def compute_personalization_score(output: AgentOutput, record: InputRecord) -> float:
    """Score how well the message uses available relevant context."""
    if not output.should_send:
        return 0.0

    available = _available_context(record)
    if not available:
        return 1.0

    blob = _message_blob(output)
    earned = 0.0
    possible = 0.0

    for key, weight in PERSONALIZATION_WEIGHTS.items():
        if key not in available:
            continue
        possible += weight
        value = available[key]
        if key == "first_name" and str(value).lower() in blob:
            earned += weight
        elif key == "property_name" and str(value).lower() in blob:
            earned += weight
        elif key == "move_date_target":
            if str(value).lower() in blob:
                earned += weight
            else:
                month = str(value)[:7]
                if month.lower() in blob:
                    earned += weight * 0.5
        elif key == "amenity_interest":
            amenities = value if isinstance(value, list) else [value]
            if any(str(a).lower() in blob for a in amenities):
                earned += weight
        elif key == "city_interest" and str(value).lower() in blob:
            earned += weight
        elif key == "cta":
            if any(term in blob for term in ("tour", "book", "schedule", "visit")):
                earned += weight

    if possible == 0:
        return 1.0
    return round(min(earned / possible, 1.0), 2)


def evaluate(
    output: AgentOutput,
    record: InputRecord,
    latency_ms: int | None = None,
) -> QualityMetrics:
    """Compute quality metrics for an agent output."""
    safety_violations = count_safety_violations(output, record)
    personalization_score = compute_personalization_score(output, record)
    return QualityMetrics(
        personalization_score=personalization_score,
        safety_violations=safety_violations,
        latency_ms=latency_ms,
    )


def check_thresholds(metrics: QualityMetrics, record: InputRecord) -> ThresholdResult:
    """Compare metrics against record thresholds if present."""
    thresholds = record.thresholds
    failures: list[str] = []

    if thresholds.personalization_score_min is not None:
        if metrics.personalization_score < thresholds.personalization_score_min:
            failures.append(
                f"personalization_score {metrics.personalization_score} < "
                f"{thresholds.personalization_score_min}"
            )

    if thresholds.safety_violations_max is not None:
        if metrics.safety_violations > thresholds.safety_violations_max:
            failures.append(
                f"safety_violations {metrics.safety_violations} > "
                f"{thresholds.safety_violations_max}"
            )

    if thresholds.p95_latency_ms is not None and metrics.latency_ms is not None:
        if metrics.latency_ms > thresholds.p95_latency_ms:
            failures.append(
                f"latency_ms {metrics.latency_ms} > {thresholds.p95_latency_ms}"
            )

    return ThresholdResult(passed=len(failures) == 0, failures=failures)
