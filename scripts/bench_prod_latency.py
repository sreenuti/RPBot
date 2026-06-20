#!/usr/bin/env python3
"""Benchmark production Vercel API latency vs direct HF."""

from __future__ import annotations

import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.loader import load_jsonl
from src.llm_client import LLMClient
from src.prompt_builder import build_prompt

PROD_URL = "https://rp-bot-lyart.vercel.app/api/run"
BATCH_SIZE = 3
MAX_RECORDS = 9


def bench_hf(records) -> list[float]:
    client = LLMClient(mock=False)
    latencies: list[float] = []
    for record in records:
        start = time.perf_counter()
        client.generate(build_prompt(record), record=record)
        latencies.append((time.perf_counter() - start) * 1000)
    return latencies


def bench_prod(records) -> tuple[list[float], list[int]]:
    latencies: list[float] = []
    llm_calls: list[int] = []
    for start in range(0, len(records), BATCH_SIZE):
        batch = [record.model_dump() for record in records[start : start + BATCH_SIZE]]
        payload = json.dumps({"records": batch, "mock": False}).encode()
        request = urllib.request.Request(
            PROD_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        batch_start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=300) as response:
            body = json.loads(response.read().decode())
        print(f"  batch wall: {round((time.perf_counter() - batch_start) * 1000)} ms")
        for record_trace in body["trace"]["records"]:
            latencies.append(record_trace["latency_ms"])
            calls = sum(1 for step in record_trace["steps"] if step["phase"] == "llm")
            llm_calls.append(calls)
            llm_ms = next(
                step["elapsed_ms"] for step in record_trace["steps"] if step["phase"] == "llm"
            )
            print(
                f"    {record_trace['task_id']}: {record_trace['latency_ms']} ms "
                f"(llm {llm_ms} ms, {calls} call(s))"
            )
    return latencies, llm_calls


def main() -> None:
    records = load_jsonl(ROOT / "data" / "test_cases.jsonl")[:MAX_RECORDS]

    print("Direct HF endpoint (this machine):")
    hf_lat = bench_hf(records[:5])
    print(f"  mean: {round(statistics.mean(hf_lat))} ms  values: {[round(x) for x in hf_lat]}")

    print(f"\nProduction {PROD_URL} ({len(records)} records, batch={BATCH_SIZE}):")
    prod_lat, prod_calls = bench_prod(records)
    print(f"  mean: {round(statistics.mean(prod_lat))} ms")
    print(f"  retries: {sum(1 for c in prod_calls if c > 1)} records with >1 LLM call")


if __name__ == "__main__":
    main()
