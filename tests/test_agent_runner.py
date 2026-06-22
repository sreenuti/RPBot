from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.agent_runner import process_record, run_batch
from src.loader import load_jsonl
from src.llm_client import LLMClient
from src.output_parser import parse_agent_output

DATA = __import__("pathlib").Path(__file__).resolve().parent.parent / "data"


def test_parse_invalid_channel_raises_validation_error():
    payload = {
        "should_send": True,
        "next_message": {
            "channel": "whatsapp",
            "send_at": "2025-12-09T09:00:00-06:00",
            "subject": None,
            "body": "Hi",
            "cta": None,
        },
        "next_action": {"type": "suppress", "details": {"reason": "test"}},
        "reasoning": "Test message.",
    }
    with pytest.raises(ValidationError):
        parse_agent_output("tc06_unknown_channel", payload)


def test_run_batch_skips_failed_records_and_continues():
    records = load_jsonl(DATA / "sample.jsonl")
    real_process = process_record

    def flaky_process(record, llm, **kwargs):
        if record.task_id == records[0].task_id:
            raise ValueError(
                "1 validation error for NextMessage\nchannel\n"
                "  Input should be 'sms', 'email', 'push' or 'none' "
                "[type=literal_error, input_value='whatsapp', input_type=str]"
            )
        return real_process(record, llm, **kwargs)

    with patch("src.agent_runner.process_record", side_effect=flaky_process):
        outputs, trace = run_batch(records, LLMClient(mock=True), capture_trace=True)

    assert len(outputs) == 1
    assert outputs[0].task_id == records[1].task_id
    assert trace is not None
    assert trace.summary.total_records == 2
    assert trace.summary.failed == 1
    failed_trace = next(r for r in trace.records if r.task_id == records[0].task_id)
    assert failed_trace.error is not None
    assert "whatsapp" in failed_trace.error
