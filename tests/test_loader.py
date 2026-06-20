import json
from pathlib import Path

import pytest

from src.loader import LoaderError, load_jsonl
from src.schemas import InputRecord

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_load_sample_jsonl():
    records = load_jsonl(DATA_DIR / "sample.jsonl")
    assert len(records) == 2
    assert records[0].task_id == "prospect_welcome_day0"
    assert records[1].input.profile.first_name == "Taylor"


def test_malformed_json_raises_loader_error(tmp_path):
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text("{not valid json\n", encoding="utf-8")
    with pytest.raises(LoaderError, match="Line 1: invalid JSON"):
        load_jsonl(bad_file)


def test_schema_error_raises_loader_error(tmp_path):
    bad_file = tmp_path / "bad_schema.jsonl"
    bad_file.write_text(json.dumps({"no_task_id": True}) + "\n", encoding="utf-8")
    with pytest.raises(LoaderError, match="Line 1: schema error"):
        load_jsonl(bad_file)


def test_skips_empty_lines(tmp_path):
    file_path = tmp_path / "sparse.jsonl"
    file_path.write_text(
        "\n" + json.dumps({"task_id": "only_one"}) + "\n\n",
        encoding="utf-8",
    )
    records = load_jsonl(file_path)
    assert len(records) == 1
    assert records[0].task_id == "only_one"
