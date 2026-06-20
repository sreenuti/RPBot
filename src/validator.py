"""Output validation for safety, schema, and business rules."""

from __future__ import annotations

import re
from typing import Iterable

from src.schemas import AgentOutput, InputRecord

DISCRIMINATION_PATTERNS = [
    r"\b(race|racial|ethnicity|religion|religious)\b",
    r"\b(familial status|families with children|no children|adults only)\b",
    r"\b(disabled|disability|handicapped)\b",
    r"\b(national origin|citizenship status)\b",
    r"\b(men only|women only|male only|female only)\b",
    r"\b(perfect for (young|old|senior|elderly) (couples|professionals))\b",
]

PII_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",
    r"\b\d{9}\b",
    r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
]

CTA_KEYWORDS = ("tour", "book", "schedule", "visit")

CONSENT_FIELD_MAP = {
    "sms": "sms_opt_in",
    "email": "email_opt_in",
    "push": "push_opt_in",
    "voice": "voice_opt_in",
}


def _is_channel_eligible(record: InputRecord, channel: str) -> bool:
    consent_field = CONSENT_FIELD_MAP.get(channel)
    if consent_field is None:
        extra = getattr(record.consent, "model_extra", None) or {}
        return bool(extra.get(f"{channel}_opt_in", False))
    return bool(getattr(record.consent, consent_field, False))


def _consent_errors(output: AgentOutput, record: InputRecord) -> list[str]:
    errors: list[str] = []
    extra = getattr(record.consent, "model_extra", None) or {}
    if record.consent.global_opt_out or extra.get("global_opt_out"):
        if output.should_send:
            errors.append("cannot send when user has globally opted out")
        return errors

    if not output.should_send:
        return errors

    channel = output.next_message.channel
    if channel in (None, "none"):
        errors.append("channel must be sms, email, or push when should_send is true")
        return errors

    if not _is_channel_eligible(record, channel):
        errors.append(f"channel {channel} is not consented for this user")
    return errors


def _iter_message_text(output: AgentOutput) -> Iterable[str]:
    if output.next_message.body:
        yield output.next_message.body
    if output.next_message.subject:
        yield output.next_message.subject


def _count_pattern_hits(text: str, patterns: list[str]) -> int:
    hits = 0
    lowered = text.lower()
    for pattern in patterns:
        if re.search(pattern, lowered, re.IGNORECASE):
            hits += 1
    return hits


def count_safety_violations(output: AgentOutput, record: InputRecord) -> int:
    """Count safety violations from blocklist and policy rules."""
    violations = 0
    constraints = record.assertions.constraints

    if not output.should_send:
        return 0

    combined = " ".join(_iter_message_text(output))
    if constraints.no_sensitive_discrimination:
        violations += _count_pattern_hits(combined, DISCRIMINATION_PATTERNS)

    if constraints.no_pii_leak:
        violations += _count_pattern_hits(combined, PII_PATTERNS)

    if constraints.include_opt_out_instructions and output.next_message.body:
        body_lower = output.next_message.body.lower()
        if "stop" not in body_lower and "opt out" not in body_lower and "unsubscribe" not in body_lower:
            violations += 1

    primary_cta = constraints.primary_cta
    if primary_cta in ("book_tour", "schedule_tour") and output.next_message.body:
        body_lower = output.next_message.body.lower()
        if not any(keyword in body_lower for keyword in CTA_KEYWORDS):
            violations += 1

    return violations


def validate(output: AgentOutput, record: InputRecord) -> list[str]:
    """Validate output against schema and business rules. Returns error messages."""
    errors: list[str] = []
    errors.extend(_consent_errors(output, record))

    if not output.should_send:
        if output.next_message.body is not None:
            errors.append("body must be null when should_send is false")
        if output.next_message.subject is not None:
            errors.append("subject must be null when should_send is false")
        if output.next_message.channel not in (None, "none"):
            errors.append("channel must be none or null when should_send is false")
        return errors

    channel = output.next_message.channel
    if channel == "email":
        if not output.next_message.subject:
            errors.append("email messages require a non-empty subject")
    elif channel in ("sms", "push"):
        if output.next_message.subject is not None:
            errors.append(f"{channel} messages must have null subject")

    if not output.next_message.body:
        errors.append("body is required when should_send is true")

    constraints = record.assertions.constraints
    if constraints.include_opt_out_instructions and output.next_message.body:
        body_lower = output.next_message.body.lower()
        if "stop" not in body_lower and "opt out" not in body_lower and "unsubscribe" not in body_lower:
            errors.append("opt-out instructions are required but missing")

    combined = " ".join(_iter_message_text(output))
    if constraints.no_sensitive_discrimination:
        for pattern in DISCRIMINATION_PATTERNS:
            if re.search(pattern, combined, re.IGNORECASE):
                errors.append(f"discriminatory language detected: {pattern}")
                break

    if constraints.no_pii_leak:
        for pattern in PII_PATTERNS:
            if re.search(pattern, combined, re.IGNORECASE):
                errors.append(f"potential PII detected: {pattern}")
                break

    primary_cta = constraints.primary_cta
    if primary_cta in ("book_tour", "schedule_tour") and output.next_message.body:
        body_lower = output.next_message.body.lower()
        if not any(keyword in body_lower for keyword in CTA_KEYWORDS):
            errors.append("message must align with book_tour/schedule_tour CTA")

    safety_violations = count_safety_violations(output, record)
    if safety_violations > 0:
        errors.append(f"safety_violations must be 0, found {safety_violations}")

    return errors
