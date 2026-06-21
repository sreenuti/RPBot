import pytest

from src.llm_client import LLMClient
from src.llm_judge import build_judge_prompt, parse_judge_response, run_llm_judge
from src.loader import load_jsonl
from src.output_parser import parse_agent_output


@pytest.fixture
def sample_record():
    records = load_jsonl(__import__("pathlib").Path("data/sample.jsonl"))
    return records[0]


@pytest.fixture
def sample_output(sample_record):
    return parse_agent_output(
        sample_record.task_id,
        {
            "should_send": True,
            "next_message": {
                "channel": "sms",
                "send_at": "2025-12-09T09:00:00-06:00",
                "body": "Hi Taylor—welcome to Oak Ridge! Reply STOP to opt out.",
                "cta": {"type": "schedule_tour"},
            },
            "next_action": {"type": "start_cadence", "details": {"name": "welcome"}},
            "reasoning": "Send welcome SMS.",
        },
    )


def test_build_judge_prompt_excludes_expected(sample_record, sample_output):
    prompt = build_judge_prompt(sample_record, sample_output)
    assert "expected" not in prompt.lower()
    assert sample_record.task_id in prompt
    assert "agent_output" in prompt


def test_parse_judge_response_normalizes_scores():
    result = parse_judge_response(
        {
            "passed": True,
            "overall_score": 0.91,
            "decision_score": 0.88,
            "compliance_score": 0.95,
            "tone_score": 0.9,
            "reasoning": "Good message.",
        }
    )
    assert result.passed is True
    assert result.overall_score == 0.91
    assert result.reasoning == "Good message."


def test_run_llm_judge_calls_complete_json(sample_record, sample_output, monkeypatch):
    client = LLMClient(mock=False, provider="openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def fake_complete_json(prompt, *, system_message=None):
        assert sample_record.task_id in prompt
        return {
            "passed": True,
            "overall_score": 0.82,
            "decision_score": 0.8,
            "compliance_score": 0.9,
            "tone_score": 0.85,
            "reasoning": "Appropriate welcome SMS with opt-out.",
        }

    monkeypatch.setattr(client, "complete_json", fake_complete_json)
    result = run_llm_judge(sample_record, sample_output, client)
    assert result.passed is True
    assert result.overall_score == 0.82
