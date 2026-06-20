from src.loader import load_jsonl
from src.output_parser import parse_agent_output
from src.output_sanitizer import sanitize_output
from src.validator import validate

DATA = __import__("pathlib").Path(__file__).resolve().parent.parent / "data"


def test_sanitize_appends_sms_opt_out():
    record = load_jsonl(DATA / "test_cases.jsonl")[0]
    output = parse_agent_output(
        record.task_id,
        {
            "should_send": True,
            "next_message": {
                "channel": "sms",
                "send_at": "2025-12-09T10:00:00-06:00",
                "subject": None,
                "body": "Hi Taylor, welcome to Oak Ridge!",
                "cta": {"type": "schedule_tour"},
            },
            "next_action": {"type": "start_cadence", "details": {"name": "welcome"}},
            "reasoning": "Send welcome SMS.",
        },
    )
    fixed = sanitize_output(output, record)
    assert "STOP" in (fixed.next_message.body or "")
    assert not validate(fixed, record)


def test_sanitize_suppresses_global_opt_out():
    record = load_jsonl(DATA / "test_cases.jsonl")[2]
    assert record.consent.global_opt_out
    output = parse_agent_output(
        record.task_id,
        {
            "should_send": True,
            "next_message": {
                "channel": "sms",
                "send_at": "2025-12-09T10:00:00-06:00",
                "subject": None,
                "body": "Hello",
                "cta": None,
            },
            "next_action": {"type": "follow_up_in_days", "details": {"value": 1}},
            "reasoning": "Should not send.",
        },
    )
    fixed = sanitize_output(output, record)
    assert fixed.should_send is False
    assert fixed.next_message.channel == "none"
    assert not validate(fixed, record)
