#!/usr/bin/env python3
"""Smoke-test a remote OpenAI-compatible model endpoint (HF, local serve, etc.)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from src.llm_client import LLMClient, LLMError
from src.loader import load_jsonl
from src.output_parser import parse_agent_output
from src.prompt_builder import build_prompt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test remote/local model endpoint")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "sample.jsonl",
        help="JSONL file; uses first record for a single prompt test",
    )
    args = parser.parse_args(argv)

    load_dotenv(ROOT / ".env")
    provider = os.getenv("LLM_PROVIDER", "openai")
    if provider != "local":
        print(
            f"WARNING: LLM_PROVIDER={provider!r}. Set LLM_PROVIDER=local for fine-tuned model.",
            file=sys.stderr,
        )

    base_url = os.getenv("LOCAL_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if not base_url:
        print("Set LOCAL_BASE_URL in .env", file=sys.stderr)
        return 1

    records = load_jsonl(args.input)
    record = records[0]
    prompt = build_prompt(record)

    print(f"Endpoint: {base_url}")
    print(f"Model: {os.getenv('LOCAL_MODEL', '(default)')}")
    print(f"Task: {record.task_id}\n")

    llm = LLMClient(mock=False)
    try:
        payload = llm.generate(prompt, record=record)
    except LLMError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    print("Raw JSON response:")
    print(json.dumps(payload, indent=2)[:3000])

    try:
        output = parse_agent_output(record.task_id, payload)
    except Exception as exc:
        print(f"\nParse warning: {exc}", file=sys.stderr)
        return 1

    print(f"\nParsed OK — should_send={output.should_send}, channel={output.next_message.channel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
