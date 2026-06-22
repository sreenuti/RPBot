import json

import pytest

from src.loader import load_jsonl
from src.prompt_builder import (
    TRAINING_SYSTEM_PROMPT,
    build_prompt,
    build_retry_prompt,
    prompt_style,
    record_to_context,
)

DATA = __import__("pathlib").Path(__file__).resolve().parent.parent / "data"


def test_record_to_context_excludes_expected_and_thresholds():
    record = load_jsonl(DATA / "sample.jsonl")[0]
    assert record.expected is not None
    assert record.thresholds is not None
    context = record_to_context(record)
    assert "expected" not in context
    assert "thresholds" not in context
    assert context["task_id"] == record.task_id


def test_build_full_prompt_contains_guidance_and_schema(monkeypatch):
    monkeypatch.setenv("PROMPT_STYLE", "full")
    record = load_jsonl(DATA / "sample.jsonl")[0]
    prompt = build_prompt(record)
    assert "autonomous property management communication agent" in prompt.lower()
    assert "should_send" in prompt
    assert "Respond with JSON only" in prompt
    assert "thresholds" not in prompt


def test_build_training_prompt_is_compact_json(monkeypatch):
    monkeypatch.setenv("PROMPT_STYLE", "training")
    record = load_jsonl(DATA / "sample.jsonl")[0]
    prompt = build_prompt(record)
    parsed = json.loads(prompt)
    assert parsed["task_id"] == record.task_id
    assert "expected" not in parsed
    assert "thresholds" not in parsed

def test_default_prompt_style_is_full(monkeypatch):
    monkeypatch.delenv("PROMPT_STYLE", raising=False)
    assert prompt_style() == "full"


def test_build_retry_prompt_includes_validation_errors(monkeypatch):
    monkeypatch.setenv("PROMPT_STYLE", "full")
    record = load_jsonl(DATA / "sample.jsonl")[0]
    errors = ["body must be null when should_send is false", "channel must be none"]
    retry = build_retry_prompt(record, errors)
    assert "Your previous response failed validation" in retry
    assert "body must be null when should_send is false" in retry
    assert build_prompt(record) in retry


def test_training_system_prompt_defined():
    assert "RealPage" in TRAINING_SYSTEM_PROMPT
