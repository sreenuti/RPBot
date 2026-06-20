import json

from src.loader import load_jsonl
from src.prompt_builder import build_prompt, build_retry_prompt, record_to_context

DATA = __import__("pathlib").Path(__file__).resolve().parent.parent / "data"


def test_record_to_context_excludes_expected():
    record = load_jsonl(DATA / "sample.jsonl")[0]
    assert record.expected is not None
    context = record_to_context(record)
    assert "expected" not in context
    assert context["task_id"] == record.task_id


def test_build_prompt_contains_guidance_and_schema():
    record = load_jsonl(DATA / "sample.jsonl")[0]
    prompt = build_prompt(record)
    assert "autonomous property management communication agent" in prompt.lower()
    assert "should_send" in prompt
    assert "Respond with JSON only" in prompt
    parsed = json.loads(prompt.split("\n\n", 1)[1])
    assert parsed["input_record"]["task_id"] == record.task_id
    assert "expected" not in parsed["input_record"]


def test_build_retry_prompt_includes_validation_errors():
    record = load_jsonl(DATA / "sample.jsonl")[0]
    errors = ["body must be null when should_send is false", "channel must be none"]
    retry = build_retry_prompt(record, errors)
    assert "Your previous response failed validation" in retry
    assert "body must be null when should_send is false" in retry
    assert build_prompt(record) in retry
