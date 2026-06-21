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
from src.llm_client import LLMClient, LLMError, _env  # noqa: E402
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
    mock: bool = False
    use_openai: bool = False
    use_judge: bool = True


def _mask_secret(value: str | None) -> str | None:
    if not value:
        return None
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}…{value[-4:]}"


def _provider_status() -> dict:
    local_url = _env("LOCAL_BASE_URL") or _env("OPENAI_BASE_URL")
    local_key = _env("LOCAL_API_KEY")
    openai_key = _env("OPENAI_API_KEY")
    hf_remote = bool(local_url and "huggingface" in local_url.lower())
    local_key_ok = bool(local_key and local_key not in ("local", ""))
    return {
        "openai": {
            "configured": bool(openai_key),
            "model": _env("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini",
            "key_preview": _mask_secret(openai_key),
        },
        "hf": {
            "configured": bool(local_url and local_key_ok),
            "base_url": local_url,
            "model": _env("LOCAL_MODEL"),
            "key_preview": _mask_secret(local_key),
            "needs_hf_token": hf_remote and not local_key_ok,
        },
    }


def _llm_for_request(*, use_openai: bool) -> LLMClient:
    """Build agent LLM client: OpenAI or HF (local) endpoint."""
    provider = "openai" if use_openai else "local"
    local_url = _env("LOCAL_BASE_URL") or _env("OPENAI_BASE_URL")
    if provider == "local" and not local_url:
        raise HTTPException(
            status_code=400,
            detail="HF endpoint is not configured. Set LOCAL_BASE_URL on the server.",
        )
    if provider == "local":
        local_key = _env("LOCAL_API_KEY")
        hf_remote = bool(local_url and "huggingface" in local_url.lower())
        if hf_remote and (not local_key or local_key in ("local", "")):
            raise HTTPException(
                status_code=400,
                detail=(
                    "HF endpoint requires LOCAL_API_KEY on the server "
                    "(your Hugging Face token, hf_…). OpenAI key cannot be used for HF."
                ),
            )
    if provider == "openai" and not _env("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=400,
            detail="OpenAI is not configured. Set OPENAI_API_KEY on the server.",
        )
    return LLMClient(mock=False, provider=provider)


def _judge_llm_for_request(*, use_judge: bool) -> LLMClient | None:
    """Separate LLM for judge; prefers OpenAI when configured."""
    if not use_judge:
        return None
    if _env("OPENAI_API_KEY"):
        return LLMClient(mock=False, provider="openai")
    local_url = _env("LOCAL_BASE_URL") or _env("OPENAI_BASE_URL")
    if not local_url:
        raise HTTPException(
            status_code=400,
            detail="LLM judge requires OPENAI_API_KEY or LOCAL_BASE_URL on the server.",
        )
    local_key = _env("LOCAL_API_KEY")
    hf_remote = bool(local_url and "huggingface" in local_url.lower())
    if hf_remote and (not local_key or local_key in ("local", "")):
        raise HTTPException(
            status_code=400,
            detail="LLM judge on HF requires LOCAL_API_KEY on the server.",
        )
    return LLMClient(mock=False, provider="local")


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


@app.get("/api/config")
async def config() -> dict:
    """Safe provider config for debugging (no secrets)."""
    return {"providers": _provider_status()}


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

    if request.mock:
        llm = LLMClient(mock=True)
        judge_llm = None
    else:
        llm = _llm_for_request(use_openai=request.use_openai)
        judge_llm = _judge_llm_for_request(use_judge=request.use_judge)
    try:
        outputs, trace = run_batch(
            records,
            llm,
            judge_llm=judge_llm,
            capture_trace=True,
        )
    except LLMError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "outputs": [output.model_dump() for output in outputs],
        "trace": trace.model_dump() if trace else None,
    }


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
