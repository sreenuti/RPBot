"""FastAPI demo UI for the RealPage message agent."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_runner import run_batch  # noqa: E402
from src.loader import LoaderError, load_jsonl  # noqa: E402
from src.llm_client import LLMClient, LLMError  # noqa: E402
from src.schemas import InputRecord  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
SAMPLE_PATH = ROOT / "data" / "sample.jsonl"

app = FastAPI(
    title="RealPage Message Agent",
    description="Interactive demo for the autonomous communication agent",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    records: list[dict]
    mock: bool = True


def _parse_jsonl_bytes(content: bytes, filename: str) -> list[dict]:
    text = content.decode("utf-8")
    records: list[dict] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LoaderError(f"Line {line_number}: invalid JSON: {exc.msg}") from exc
        try:
            record = InputRecord.model_validate(payload)
        except ValidationError as exc:
            raise LoaderError(f"Line {line_number}: schema error: {exc}") from exc
        records.append(record.model_dump())
    if not records:
        raise LoaderError(f"No records found in {filename}")
    return records


@app.get("/")
async def index() -> FileResponse:
    response = FileResponse(STATIC_DIR / "index.html")
    response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/sample")
async def get_sample() -> dict:
    try:
        records = load_jsonl(SAMPLE_PATH)
    except LoaderError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "filename": "sample.jsonl",
        "record_count": len(records),
        "records": [record.model_dump() for record in records],
    }


@app.post("/api/upload")
async def upload_jsonl(file: UploadFile = File(...)) -> dict:
    if not file.filename or not file.filename.lower().endswith(".jsonl"):
        raise HTTPException(status_code=400, detail="Upload a .jsonl file")

    content = await file.read()
    try:
        records = _parse_jsonl_bytes(content, file.filename)
    except LoaderError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "filename": file.filename,
        "record_count": len(records),
        "records": records,
    }


@app.post("/api/run")
async def run_agent(request: RunRequest) -> dict:
    if not request.records:
        raise HTTPException(status_code=400, detail="No records to process")

    try:
        records = [InputRecord.model_validate(item) for item in request.records]
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    llm = LLMClient(mock=request.mock)
    try:
        outputs, trace = run_batch(records, llm, capture_trace=True)
    except LLMError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "outputs": [output.model_dump() for output in outputs],
        "trace": trace.model_dump() if trace else None,
    }


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
