"""Export agent outputs to JSONL and console."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from src.schemas import AgentOutput


def _safe_print(text: str) -> None:
    """Print safely on Windows consoles that lack full Unicode support."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode(encoding, errors="replace").decode(encoding))


def export_jsonl(outputs: list[AgentOutput], path: str | Path) -> None:
    """Write one JSON object per line to the output file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for output in outputs:
            handle.write(output.model_dump_json())
            handle.write("\n")


def print_record_result(output: AgentOutput) -> None:
    """Print a copy-paste friendly per-record result block."""
    channel = output.next_message.channel or "none"
    lines = [
        "-" * 60,
        f"task_id: {output.task_id}",
        f"should_send: {output.should_send}",
        f"channel: {channel}",
        f"send_at: {output.next_message.send_at}",
    ]
    if output.next_message.subject:
        lines.append(f"subject: {output.next_message.subject}")
    if output.next_message.body:
        lines.append(f"body: {output.next_message.body}")
    lines.append(
        f"next_action: {output.next_action.type} {output.next_action.details or ''}".strip()
    )
    lines.append(f"reasoning: {output.reasoning}")
    lines.append(
        "quality: "
        f"personalization={output.quality.personalization_score}, "
        f"safety_violations={output.quality.safety_violations}, "
        f"latency_ms={output.quality.latency_ms}"
    )
    for line in lines:
        _safe_print(line)


def print_summary(outputs: list[AgentOutput]) -> None:
    """Print aggregate run summary."""
    total = len(outputs)
    sent = sum(1 for o in outputs if o.should_send)
    suppressed = total - sent
    channels = Counter(
        (o.next_message.channel or "none") if o.should_send else "suppressed"
        for o in outputs
    )
    scores = [o.quality.personalization_score for o in outputs]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    max_safety = max((o.quality.safety_violations for o in outputs), default=0)

    summary_lines = [
        "=" * 60,
        "RUN SUMMARY",
        "=" * 60,
        f"total records: {total}",
        f"sent: {sent}",
        f"suppressed: {suppressed}",
        "channel distribution:",
    ]
    for channel, count in sorted(channels.items()):
        summary_lines.append(f"  {channel}: {count}")
    summary_lines.extend(
        [
            f"average personalization score: {avg_score:.2f}",
            f"max safety violations: {max_safety}",
        ]
    )
    for line in summary_lines:
        _safe_print(line)
