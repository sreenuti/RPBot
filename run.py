#!/usr/bin/env python3
"""CLI entrypoint for the RealPage message agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.agent_runner import process_record, run_batch
from src.exporter import export_jsonl, print_record_result, print_summary
from src.loader import LoaderError, load_jsonl
from src.llm_client import LLMClient, LLMError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RealPage context-aware message agent")
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock LLM")
    parser.add_argument("--verbose", action="store_true", help="Print detailed per-record output")
    args = parser.parse_args(argv)

    try:
        records = load_jsonl(args.input)
    except LoaderError as exc:
        print(f"Error loading input: {exc}", file=sys.stderr)
        return 1

    llm = LLMClient(mock=args.mock)

    try:
        outputs, _ = run_batch(records, llm, verbose=args.verbose)
    except LLMError as exc:
        print(f"Error processing records: {exc}", file=sys.stderr)
        return 1

    if args.verbose:
        for output in outputs:
            print_record_result(output)

    export_jsonl(outputs, Path(args.output))
    print_summary(outputs)
    print(f"\nWrote {len(outputs)} record(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
