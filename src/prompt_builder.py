"""Build LLM prompts for fully autonomous communication decisions."""

from __future__ import annotations

import json

from src.schemas import InputRecord

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


def record_to_context(record: InputRecord) -> dict:
    """Serialize the input record for the LLM, excluding hold-out expected outputs."""
    return record.model_dump(exclude={"expected"})


def build_prompt(record: InputRecord) -> str:
    """Construct a prompt for a fully autonomous communication decision."""
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
        f"{json.dumps(instructions, indent=2)}"
    )


def build_retry_prompt(record: InputRecord, validation_errors: list[str]) -> str:
    """Append validation feedback so the LLM can correct its prior response."""
    feedback = "\n".join(f"- {error}" for error in validation_errors)
    return (
        f"{build_prompt(record)}\n\n"
        "Your previous response failed validation:\n"
        f"{feedback}\n\n"
        "Fix all issues and respond with valid JSON only."
    )
