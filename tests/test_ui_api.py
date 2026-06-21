import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.app import app  # noqa: E402

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_sample_returns_records():
    response = client.get("/api/sample")
    assert response.status_code == 200
    data = response.json()
    assert data["record_count"] == 2
    assert len(data["records"]) == 2
    assert data["records"][0]["task_id"] == "prospect_welcome_day0"


def test_run_mock_agent():
    sample = client.get("/api/sample").json()
    response = client.post("/api/run", json={"records": sample["records"], "mock": True})
    assert response.status_code == 200
    data = response.json()
    assert len(data["outputs"]) == 2
    assert data["trace"] is not None
    assert data["trace"]["mock"] is True
    assert data["trace"]["summary"]["total_records"] == 2


def test_run_accepts_use_openai_flag_with_mock():
    sample = client.get("/api/sample").json()
    response = client.post(
        "/api/run",
        json={"records": sample["records"], "mock": True, "use_openai": True},
    )
    assert response.status_code == 200


def test_run_rejects_empty_records():
    response = client.post("/api/run", json={"records": [], "mock": True})
    assert response.status_code == 400


def test_upload_rejects_non_jsonl():
    response = client.post(
        "/api/upload",
        files={"file": ("data.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


def test_upload_valid_jsonl(tmp_path):
    jsonl = ROOT / "data" / "sample.jsonl"
    response = client.post(
        "/api/upload",
        files={"file": ("sample.jsonl", jsonl.read_bytes(), "application/jsonl")},
    )
    assert response.status_code == 200
    assert response.json()["record_count"] == 2
