"""Build LLM prompts for fully autonomous communication decisions."""

from __future__ import annotations

import json
import os

from src.schemas import InputRecord

TRAINING_SYSTEM_PROMPT = (
    "You are a RealPage communication decision agent. "
    "Return only valid JSON matching the expected schema."
)

OUTPUT_SCHEMA = {
    "should_send": "boolean — whether to send a message now",
    "next_message": {
        "channel": "sms | email | push | none",
        "send_at": "ISO-8601 datetime with timezone offset, or null if should_send is false",
        "subject": "string for email, null for sms/push/none",
        "body": "string message body, or null if should_send is false",
        "cta": "object describing the call-to-action, or null if should_send is false",
    },
    "next_action": {
        "type": "string e.g. start_cadence, follow_up_in_days, suppress",
        "details": "object with action-specific fields, or omit if not applicable",
    },
    "reasoning": "brief explanation of your decisions",
}


def prompt_style() -> str:
    """Return ``full`` (default), ``compact``, or ``training`` (experimental)."""
    return os.getenv("PROMPT_STYLE", "full").lower()


def record_to_context(record: InputRecord) -> dict:
    """Serialize the input record for the LLM, excluding hold-out and eval-only fields."""
    return record.model_dump(exclude={"expected", "thresholds"})


def _training_prompt(record: InputRecord) -> str:
    """Compact JSON user message — matches LoRA chat rows (experimental at inference)."""
    return json.dumps(record_to_context(record), ensure_ascii=False, separators=(",", ":"))


def _compact_prompt(record: InputRecord) -> str:
    """Shorter prompt that still declares required output keys."""
    instructions = {
        "input_record": record_to_context(record),
        "output_keys": ["should_send", "next_message", "next_action", "reasoning"],
        "rules": [
            "Respect consent and channel_preferences order.",
            "Global opt-out or no consented channel => should_send false, channel none.",
            "Honor assertions.constraints (opt-out, primary_cta, quiet_hours, send_at).",
            "Use input timezone for send_at.",
            "Personalize; avoid fair housing violations and unnecessary PII.",
        ],
    }
    return "Respond with JSON only.\n" + json.dumps(instructions, ensure_ascii=False, separators=(",", ":"))


def _full_prompt(record: InputRecord) -> str:
    """Full schema prompt with compact JSON serialization (default production path)."""
    instructions = {
        "role": "autonomous_property_management_communication_agent",
        "task": (
            "Analyze the input record and decide the complete next communication: "
            "whether to send, which channel, when, message content, CTA, and follow-up action."
        ),
        "input_record": record_to_context(record),
        "output_schema": OUTPUT_SCHEMA,
        "guidance": [
            "Decide entirely from input_record fields. Never use or infer from any expected/gold output.",
            "Respect consent flags and channel_preferences order when choosing a channel.",
            "If the user has globally opted out or no channel is consented, set should_send to false.",
            "Honor assertions.constraints (opt-out language, primary_cta, quiet_hours, send_at overrides).",
            "Use input timezone for send_at. Consider lifecycle_stage, last_interaction, and move_date_target.",
            "Personalize using profile and property context. Avoid Fair Housing violations and unnecessary PII.",
            "When should_send is false: channel must be none, body/subject/cta must be null.",
            "Return valid JSON only with keys: should_send, next_message, next_action, reasoning.",
        ],
    }
    return (
        "You are an autonomous property management communication agent.\n"
        "Respond with JSON only. No markdown fences.\n\n"
        f"{json.dumps(instructions, ensure_ascii=False, separators=(',', ':'))}"
    )


def _resolve_style(*, for_local: bool = False) -> str:
    """Pick prompt style; HF/local fine-tune uses training format by default."""
    style = prompt_style()
    if for_local and style == "full":
        return "training"
    return style


def build_prompt(record: InputRecord, *, for_local: bool = False) -> str:
    """Construct a prompt for a fully autonomous communication decision."""
    style = _resolve_style(for_local=for_local)
    if style == "training":
        return _training_prompt(record)
    if style == "compact":
        return _compact_prompt(record)
    return _full_prompt(record)


def build_retry_prompt(
    record: InputRecord,
    validation_errors: list[str],
    *,
    for_local: bool = False,
) -> str:
    """Append validation feedback so the LLM can correct its prior response."""
    feedback = "\n".join(f"- {error}" for error in validation_errors)
    return (
        f"{build_prompt(record, for_local=for_local)}\n\n"
        "Your previous response failed validation:\n"
        f"{feedback}\n\n"
        "Fix all issues and respond with valid JSON only."
    )
