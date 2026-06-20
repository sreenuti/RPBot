from src.loader import load_jsonl
from src.llm_client import _mock_autonomous_decision
from src.output_parser import OutputParseError, parse_agent_output
from src.schemas import Consent, InputRecord
from src.validator import validate

DATA = __import__("pathlib").Path(__file__).resolve().parent.parent / "data"


def _record(**overrides) -> InputRecord:
    base = load_jsonl(DATA / "sample.jsonl")[0]
    data = base.model_dump()
    data.update(overrides)
    return InputRecord.model_validate(data)


def test_parse_agent_output_full_payload():
    payload = {
        "should_send": True,
        "next_message": {
            "channel": "sms",
            "send_at": "2025-12-09T09:00:00-06:00",
            "subject": None,
            "body": "Hi Taylor, book a tour. Reply STOP to opt out.",
            "cta": {"type": "schedule_tour", "options": ["Thu", "Fri"]},
        },
        "next_action": {
            "type": "start_cadence",
            "details": {"name": "prospect_welcome_short_horizon"},
        },
        "reasoning": "Welcome SMS with tour CTA.",
    }
    output = parse_agent_output("test_task", payload)
    assert output.should_send is True
    assert output.next_message.channel == "sms"
    assert output.next_action.type == "start_cadence"


def test_parse_agent_output_flat_next_action():
    payload = {
        "should_send": True,
        "next_message": {
            "channel": "email",
            "send_at": "2025-12-09T10:00:00-06:00",
            "subject": "Tour Oak Ridge",
            "body": "Book a tour. Reply STOP to opt out.",
            "cta": {"type": "schedule_tour"},
        },
        "next_action": {"type": "follow_up_in_days", "value": 3},
        "reasoning": "Email follow-up.",
    }
    output = parse_agent_output("test_task", payload)
    assert output.next_action.details == {"value": 3}


def test_parse_agent_output_missing_field_raises():
    try:
        parse_agent_output("test_task", {"should_send": True})
    except OutputParseError:
        pass
    else:
        raise AssertionError("Expected OutputParseError")


def test_mock_autonomous_sms_preferred():
    record = _record()
    payload = _mock_autonomous_decision(record)
    output = parse_agent_output(record.task_id, payload)
    assert output.should_send is True
    assert output.next_message.channel == "sms"
    assert validate(output, record) == []


def test_mock_autonomous_email_when_sms_opted_out():
    record = load_jsonl(DATA / "sample.jsonl")[1]
    payload = _mock_autonomous_decision(record)
    output = parse_agent_output(record.task_id, payload)
    assert output.should_send is True
    assert output.next_message.channel == "email"


def test_mock_autonomous_suppresses_on_global_opt_out():
    record = _record(consent=Consent(email_opt_in=True, sms_opt_in=True, global_opt_out=True))
    payload = _mock_autonomous_decision(record)
    output = parse_agent_output(record.task_id, payload)
    assert output.should_send is False
    assert output.next_message.channel == "none"


def test_mock_autonomous_suppresses_when_no_consented_channel():
    record = _record(
        consent=Consent(email_opt_in=False, sms_opt_in=False, voice_opt_in=False),
        channel_preferences=["sms", "email"],
    )
    payload = _mock_autonomous_decision(record)
    output = parse_agent_output(record.task_id, payload)
    assert output.should_send is False
