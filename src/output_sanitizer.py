"""Deterministic post-parse fixes to avoid expensive LLM validation retries."""

from __future__ import annotations

from src.schemas import AgentOutput, InputRecord, NextAction, NextMessage

CTA_KEYWORDS = ("tour", "book", "schedule", "visit")
OPT_OUT_MARKERS = ("stop", "opt out", "unsubscribe")


def _has_opt_out(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in OPT_OUT_MARKERS)


def _opt_out_suffix(channel: str | None) -> str | None:
    if channel == "sms":
        return " Reply STOP to opt out."
    if channel == "email":
        return "\nTo opt out, reply STOP or unsubscribe."
    if channel == "push":
        return " Reply STOP to opt out."
    return None


def _cta_suffix(primary_cta: str | None) -> str | None:
    if primary_cta in ("book_tour", "schedule_tour"):
        return " Book a tour today."
    return None


def sanitize_output(output: AgentOutput, record: InputRecord) -> AgentOutput:
    """Apply cheap policy fixes before validation (avoids LLM retry loops)."""
    extra = getattr(record.consent, "model_extra", None) or {}
    if record.consent.global_opt_out or extra.get("global_opt_out"):
        if output.should_send:
            return AgentOutput(
                task_id=output.task_id,
                should_send=False,
                next_message=NextMessage(
                    channel="none",
                    send_at=None,
                    subject=None,
                    body=None,
                    cta=None,
                ),
                next_action=NextAction(type="suppress", details={"reason": "global_opt_out"}),
                reasoning=output.reasoning or "Suppressed due to global opt-out.",
            )
        return output

    if not output.should_send or not output.next_message.body:
        return output

    constraints = record.assertions.constraints
    body = output.next_message.body
    channel = output.next_message.channel
    updates: dict[str, str] = {}

    if constraints.include_opt_out_instructions and not _has_opt_out(body):
        suffix = _opt_out_suffix(channel)
        if suffix:
            updates["body"] = body.rstrip() + suffix

    body_for_cta = updates.get("body", body)
    primary_cta = constraints.primary_cta
    if primary_cta in ("book_tour", "schedule_tour"):
        lowered = body_for_cta.lower()
        if not any(keyword in lowered for keyword in CTA_KEYWORDS):
            suffix = _cta_suffix(primary_cta)
            if suffix:
                updates["body"] = body_for_cta.rstrip() + suffix

    if not updates:
        return output

    return output.model_copy(
        update={"next_message": output.next_message.model_copy(update=updates)}
    )
