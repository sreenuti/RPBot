"""Export labeled JSONL records into fine-tuning datasets for custom models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from src.loader import load_jsonl
from src.output_parser import OutputParseError, parse_agent_output
from src.prompt_builder import build_prompt
from src.schemas import InputRecord

FinetuneFormat = Literal["openai", "hf", "prompt_completion"]

SYSTEM_PROMPT = "Respond with JSON only."


def _normalize_next_action(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("next_action must be an object")
    action_type = raw.get("type")
    if not action_type:
        raise ValueError("next_action.type is required")
    details = raw.get("details")
    if details is None:
        details = {key: value for key, value in raw.items() if key != "type"} or None
    normalized: dict[str, Any] = {"type": str(action_type)}
    if details is not None:
        normalized["details"] = details
    return normalized


def _infer_should_send(expected: dict[str, Any]) -> bool:
    if "should_send" in expected:
        return bool(expected["should_send"])
    message = expected.get("next_message")
    if isinstance(message, dict):
        channel = message.get("channel")
        if channel in (None, "none"):
            return False
        body = message.get("body")
        return bool(body)
    return False


def _synthesize_reasoning(record: InputRecord, target: dict[str, Any]) -> str:
    if not target.get("should_send"):
        action_type = target.get("next_action", {}).get("type", "suppress")
        return (
            f"No message should be sent for this record ({action_type}) based on "
            "consent, channel preferences, and constraints."
        )

    message = target.get("next_message", {})
    channel = message.get("channel", "unknown")
    send_at = message.get("send_at") or "the scheduled time"
    persona = record.persona or "prospect"
    property_name = record.input.property_name or "the property"
    return (
        f"Send a personalized {channel} message to the {persona} for {property_name} "
        f"at {send_at}, honoring consent, channel order, and assertion constraints."
    )


def build_training_target(record: InputRecord) -> dict[str, Any]:
    """Build the assistant JSON label from a record's hold-out expected output."""
    if not record.expected:
        raise ValueError(f"{record.task_id} has no expected output")

    expected = record.expected
    if "next_message" not in expected or "next_action" not in expected:
        raise ValueError(f"{record.task_id} expected output is missing next_message or next_action")

    target: dict[str, Any] = {
        "should_send": _infer_should_send(expected),
        "next_message": expected["next_message"],
        "next_action": _normalize_next_action(expected["next_action"]),
    }
    target["reasoning"] = expected.get("reasoning") or _synthesize_reasoning(record, target)

    parse_agent_output(record.task_id, target)
    return target


def build_training_example(record: InputRecord) -> dict[str, Any]:
    """Create one training example with prompt and validated target JSON."""
    prompt = build_prompt(record)
    target = build_training_target(record)
    return {
        "task_id": record.task_id,
        "prompt": prompt,
        "completion": json.dumps(target, ensure_ascii=False),
        "target": target,
    }


def format_example(example: dict[str, Any], fmt: FinetuneFormat) -> dict[str, Any]:
    """Convert a training example into a provider-specific JSONL row."""
    if fmt == "prompt_completion":
        return {
            "prompt": example["prompt"],
            "completion": example["completion"],
            "task_id": example["task_id"],
        }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["completion"]},
    ]
    row: dict[str, Any] = {"messages": messages, "task_id": example["task_id"]}
    if fmt == "openai":
        return row
    return row


def export_finetune_jsonl(
    input_paths: list[str | Path],
    output_path: str | Path,
    *,
    fmt: FinetuneFormat = "openai",
) -> dict[str, int]:
    """Write fine-tuning JSONL from one or more labeled input files."""
    records: list[InputRecord] = []
    for path in input_paths:
        records.extend(load_jsonl(path))

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    exported = 0
    skipped = 0
    with output_file.open("w", encoding="utf-8") as handle:
        for record in records:
            try:
                example = build_training_example(record)
            except (ValueError, OutputParseError):
                skipped += 1
                continue
            handle.write(json.dumps(format_example(example, fmt), ensure_ascii=False))
            handle.write("\n")
            exported += 1

    return {"exported": exported, "skipped": skipped, "total": len(records)}
