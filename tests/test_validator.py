from src.evaluator import evaluate
from src.loader import load_jsonl
from src.schemas import AgentOutput, Consent, InputRecord, NextAction, NextMessage, QualityMetrics
from src.validator import validate

DATA = __import__("pathlib").Path(__file__).resolve().parent.parent / "data"


def _record(**overrides) -> InputRecord:
    base = load_jsonl(DATA / "sample.jsonl")[0]
    data = base.model_dump()
    data.update(overrides)
    return InputRecord.model_validate(data)


def _base_output(**kwargs) -> AgentOutput:
    defaults = {
        "task_id": "test",
        "should_send": True,
        "next_message": NextMessage(
            channel="email",
            send_at="2025-12-09T10:00:00-06:00",
            subject="Tour our community",
            body="Hi Taylor, book a tour this week. To opt out, reply STOP.",
            cta={"type": "schedule_tour"},
        ),
        "next_action": NextAction(type="follow_up_in_days", details={"value": 3}),
        "reasoning": "test",
        "quality": QualityMetrics(),
    }
    defaults.update(kwargs)
    return AgentOutput(**defaults)


def test_email_requires_subject():
    record = load_jsonl(DATA / "sample.jsonl")[1]
    output = _base_output(
        next_message=NextMessage(
            channel="email",
            send_at="2025-12-09T10:00:00-06:00",
            subject=None,
            body="Book a tour today. Reply STOP to opt out.",
        )
    )
    errors = validate(output, record)
    assert any("subject" in e for e in errors)


def test_sms_subject_must_be_null():
    record = load_jsonl(DATA / "sample.jsonl")[0]
    output = _base_output(
        next_message=NextMessage(
            channel="sms",
            send_at="2025-12-09T09:00:00-06:00",
            subject="Should not exist",
            body="Hi Taylor, book a tour. Reply STOP to opt out.",
        )
    )
    errors = validate(output, record)
    assert any("null subject" in e for e in errors)


def test_opt_out_required():
    record = load_jsonl(DATA / "sample.jsonl")[0]
    output = _base_output(
        next_message=NextMessage(
            channel="sms",
            send_at="2025-12-09T09:00:00-06:00",
            subject=None,
            body="Hi Taylor, book a tour this week.",
        )
    )
    errors = validate(output, record)
    assert any("opt-out" in e for e in errors)


def test_body_null_when_suppressed():
    record = load_jsonl(DATA / "sample.jsonl")[0]
    output = _base_output(
        should_send=False,
        next_message=NextMessage(channel="none", body="leftover", subject="x"),
    )
    errors = validate(output, record)
    assert any("body must be null" in e for e in errors)
    assert any("subject must be null" in e for e in errors)


def test_blocklist_discrimination():
    record = load_jsonl(DATA / "sample.jsonl")[0]
    output = _base_output(
        next_message=NextMessage(
            channel="sms",
            send_at="2025-12-09T09:00:00-06:00",
            subject=None,
            body="Perfect for young professionals only. Book a tour. Reply STOP to opt out.",
        )
    )
    errors = validate(output, record)
    assert any("discriminatory" in e or "safety_violations" in e for e in errors)
    metrics = evaluate(output, record, latency_ms=10)
    assert metrics.safety_violations > 0


def test_cta_alignment():
    record = load_jsonl(DATA / "sample.jsonl")[0]
    output = _base_output(
        next_message=NextMessage(
            channel="sms",
            send_at="2025-12-09T09:00:00-06:00",
            subject=None,
            body="Hello Taylor, hope you are well. Reply STOP to opt out.",
        )
    )
    errors = validate(output, record)
    assert any("book_tour" in e or "safety_violations" in e for e in errors)


def test_rejects_send_on_global_opt_out():
    record = _record(
        consent=Consent(email_opt_in=True, sms_opt_in=True, global_opt_out=True)
    )
    output = _base_output(should_send=True)
    errors = validate(output, record)
    assert any("globally opted out" in e for e in errors)


def test_rejects_unconsented_channel():
    record = load_jsonl(DATA / "sample.jsonl")[1]
    output = _base_output(
        next_message=NextMessage(
            channel="sms",
            send_at="2025-12-09T09:00:00-06:00",
            subject=None,
            body="Hi Taylor, book a tour. Reply STOP to opt out.",
        )
    )
    errors = validate(output, record)
    assert any("not consented" in e for e in errors)
