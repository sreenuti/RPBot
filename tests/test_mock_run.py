import json
import subprocess
import sys
from pathlib import Path

from run import main
from src.loader import load_jsonl
from src.schemas import AgentOutput

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "sample.jsonl"


def test_mock_run_main_produces_valid_output(tmp_path):
    output_path = tmp_path / "outputs.jsonl"
    exit_code = main(
        ["--input", str(DATA), "--output", str(output_path), "--mock"]
    )
    assert exit_code == 0
    assert output_path.exists()

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    for line in lines:
        payload = json.loads(line)
        output = AgentOutput.model_validate(payload)
        assert output.task_id
        assert isinstance(output.should_send, bool)
        assert output.reasoning


def test_mock_run_subprocess(tmp_path):
    output_path = tmp_path / "out.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run.py"),
            "--input",
            str(DATA),
            "--output",
            str(output_path),
            "--mock",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert output_path.exists()

    records = load_jsonl(DATA)
    outputs = [
        AgentOutput.model_validate(json.loads(line))
        for line in output_path.read_text(encoding="utf-8").strip().splitlines()
    ]

    welcome = next(o for o in outputs if o.task_id == "prospect_welcome_day0")
    followup = next(o for o in outputs if o.task_id == "prospect_long_horizon_day3")

    assert welcome.should_send is True
    assert welcome.next_message.channel == "sms"
    assert welcome.next_message.subject is None
    assert "STOP" in (welcome.next_message.body or "")

    assert followup.should_send is True
    assert followup.next_message.channel == "email"
    assert followup.next_message.subject
    assert followup.next_message.body

    assert len(outputs) == len(records)
