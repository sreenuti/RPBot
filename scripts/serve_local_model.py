#!/usr/bin/env python3
"""Serve a merged Hugging Face model via an OpenAI-compatible HTTP API."""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = ROOT / "models" / "realpage-message-agent-v1"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_MODEL_NAME = "realpage-message-agent-v1"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = 0.0
    max_tokens: int | None = 512
    response_format: dict[str, Any] | None = None


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "local"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelCard]


def build_app(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    *,
    served_name: str,
) -> FastAPI:
    app = FastAPI(title="RealPage Local Model Server")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": served_name}

    @app.get("/v1/models")
    def list_models() -> ModelsResponse:
        return ModelsResponse(data=[ModelCard(id=served_name)])

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
        if request.model != served_name:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown model '{request.model}'. Use '{served_name}'.",
            )

        messages = [message.model_dump() for message in request.messages]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        max_new_tokens = request.max_tokens or 512

        started = time.perf_counter()
        model.eval()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        generated = output_ids[0, inputs["input_ids"].shape[-1] :]
        content = tokenizer.decode(generated, skip_special_tokens=True).strip()

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": served_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": int(inputs["input_ids"].shape[-1]),
                "completion_tokens": int(generated.shape[-1]),
                "total_tokens": int(output_ids.shape[-1]),
            },
            "latency_ms": elapsed_ms,
        }

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Serve a local merged model with an OpenAI-compatible API"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Path to merged model directory",
    )
    parser.add_argument(
        "--served-model-name",
        default=DEFAULT_MODEL_NAME,
        help="Model name exposed to OpenAI-compatible clients",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load model in 4-bit (recommended on 6GB GPUs; much lower RAM/VRAM use)",
    )
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU is required for local serving.", file=sys.stderr)
        return 1

    model_dir = args.model_dir.resolve()
    if not (model_dir / "config.json").exists():
        print(f"ERROR: No model config found in {model_dir}", file=sys.stderr)
        return 1

    print(f"Loading model from {model_dir}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if args.load_in_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        print("Loading model in 4-bit mode (lower memory). Use --no-load-in-4bit for fp16.")
    else:
        load_kwargs["dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_dir, **load_kwargs)

    app = build_app(model, tokenizer, served_name=args.served_model_name)
    print(
        f"Serving {args.served_model_name} at http://{args.host}:{args.port}/v1\n"
        "Set in .env:\n"
        "  LLM_PROVIDER=local\n"
        f"  LOCAL_BASE_URL=http://{args.host}:{args.port}/v1\n"
        f"  LOCAL_MODEL={args.served_model_name}\n"
        "  LOCAL_JSON_MODE=true"
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
