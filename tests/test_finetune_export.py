import json
from pathlib import Path

from src.finetune_export import (
    build_training_example,
    build_training_target,
    export_finetune_jsonl,
    format_example,
)
from src.loader import load_jsonl
from src.output_parser import parse_agent_output

DATA = Path(__file__).resolve().parent.parent / "data"


def test_build_training_target_from_test_case():
    record = load_jsonl(DATA / "test_cases.jsonl")[0]
    target = build_training_target(record)
    assert target["should_send"] is True
    assert target["next_message"]["channel"] == "sms"
    assert target["next_action"]["type"] == "start_cadence"
    assert target["reasoning"]
    parse_agent_output(record.task_id, target)


def test_build_training_target_infers_should_send_from_sample():
    record = load_jsonl(DATA / "sample.jsonl")[0]
    target = build_training_target(record)
    assert target["should_send"] is True
    parse_agent_output(record.task_id, target)


def test_format_openai_messages():
    record = load_jsonl(DATA / "test_cases.jsonl")[2]
    example = build_training_example(record)
    row = format_example(example, "openai")
    assert row["messages"][0]["role"] == "system"
    assert row["messages"][1]["role"] == "user"
    assistant = json.loads(row["messages"][2]["content"])
    assert assistant["should_send"] is False


def test_export_finetune_jsonl(tmp_path):
    output_path = tmp_path / "train.jsonl"
    stats = export_finetune_jsonl(
        [DATA / "sample.jsonl", DATA / "test_cases.jsonl"],
        output_path,
        fmt="prompt_completion",
    )
    assert stats["exported"] == 22
    assert stats["skipped"] == 0
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 22
    first = json.loads(lines[0])
    assert "prompt" in first
    assert "completion" in first
