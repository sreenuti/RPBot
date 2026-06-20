"""Parse LLM JSON responses into validated AgentOutput objects."""

from __future__ import annotations

from typing import Any

from src.schemas import AgentOutput, NextAction, NextMessage


class OutputParseError(Exception):
    """Raised when LLM output cannot be parsed into AgentOutput."""


def _parse_next_action(raw: Any) -> NextAction:
    if not isinstance(raw, dict):
        raise OutputParseError("next_action must be an object")
    action_type = raw.get("type")
    if not action_type:
        raise OutputParseError("next_action.type is required")
    details = raw.get("details")
    if details is None:
        details = {k: v for k, v in raw.items() if k != "type"} or None
    return NextAction(type=str(action_type), details=details)


def _parse_next_message(raw: Any) -> NextMessage:
    if not isinstance(raw, dict):
        raise OutputParseError("next_message must be an object")
    return NextMessage(
        channel=raw.get("channel"),
        send_at=raw.get("send_at"),
        subject=raw.get("subject"),
        body=raw.get("body"),
        cta=raw.get("cta"),
    )


def parse_agent_output(task_id: str, payload: dict[str, Any]) -> AgentOutput:
    """Convert a parsed LLM JSON object into AgentOutput."""
    if "should_send" not in payload:
        raise OutputParseError("should_send is required")
    if "reasoning" not in payload:
        raise OutputParseError("reasoning is required")
    if "next_message" not in payload:
        raise OutputParseError("next_message is required")
    if "next_action" not in payload:
        raise OutputParseError("next_action is required")

    reasoning = payload.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise OutputParseError("reasoning must be a non-empty string")

    return AgentOutput(
        task_id=task_id,
        should_send=bool(payload["should_send"]),
        next_message=_parse_next_message(payload["next_message"]),
        next_action=_parse_next_action(payload["next_action"]),
        reasoning=reasoning.strip(),
    )
