from src.evaluator import check_thresholds, compute_personalization_score, evaluate
from src.loader import load_jsonl
from src.schemas import AgentOutput, NextAction, NextMessage, QualityMetrics, Thresholds

DATA = __import__("pathlib").Path(__file__).resolve().parent.parent / "data"


def _output(**kwargs) -> AgentOutput:
    defaults = {
        "task_id": "test",
        "should_send": True,
        "next_message": NextMessage(
            channel="sms",
            send_at="2025-12-09T09:00:00-06:00",
            subject=None,
            body="Hi Taylor, welcome to Oak Ridge Apartments in Richardson, TX! Book a tour. Reply STOP.",
            cta={"type": "schedule_tour"},
        ),
        "next_action": NextAction(type="start_cadence", details={"name": "welcome"}),
        "reasoning": "test",
        "quality": QualityMetrics(),
    }
    defaults.update(kwargs)
    return AgentOutput(**defaults)


def test_personalization_score_uses_available_context():
    record = load_jsonl(DATA / "sample.jsonl")[0]
    score = compute_personalization_score(_output(), record)
    assert score >= 0.8


def test_suppressed_send_scores_zero():
    record = load_jsonl(DATA / "sample.jsonl")[0]
    output = _output(
        should_send=False,
        next_message=NextMessage(channel="none", body=None, subject=None),
    )
    assert compute_personalization_score(output, record) == 0.0


def test_check_thresholds_passes_when_met():
    record = load_jsonl(DATA / "sample.jsonl")[0]
    metrics = QualityMetrics(personalization_score=0.9, safety_violations=0, latency_ms=100)
    result = check_thresholds(metrics, record)
    assert result.passed is True
    assert result.failures == []


def test_check_thresholds_fails_on_low_personalization():
    record = load_jsonl(DATA / "sample.jsonl")[0]
    metrics = QualityMetrics(personalization_score=0.5, safety_violations=0, latency_ms=100)
    result = check_thresholds(metrics, record)
    assert result.passed is False
    assert any("personalization_score" in f for f in result.failures)


def test_check_thresholds_fails_on_latency():
    record = load_jsonl(DATA / "sample.jsonl")[0]
    custom = record.model_copy(update={"thresholds": Thresholds(p95_latency_ms=50)})
    metrics = QualityMetrics(personalization_score=1.0, safety_violations=0, latency_ms=200)
    result = check_thresholds(metrics, custom)
    assert result.passed is False
    assert any("latency_ms" in f for f in result.failures)


def test_evaluate_computes_all_metrics():
    record = load_jsonl(DATA / "sample.jsonl")[0]
    metrics = evaluate(_output(), record, latency_ms=42)
    assert metrics.latency_ms == 42
    assert metrics.safety_violations == 0
    assert metrics.personalization_score > 0
