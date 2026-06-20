#!/usr/bin/env python3
"""Export labeled JSONL records into fine-tuning datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.finetune_export import export_finetune_jsonl  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export fine-tuning JSONL from labeled input records"
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more labeled JSONL files with expected outputs",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write fine-tuning JSONL",
    )
    parser.add_argument(
        "--format",
        choices=("openai", "hf", "prompt_completion"),
        default="openai",
        help="Output format (default: openai chat JSONL)",
    )
    args = parser.parse_args(argv)

    stats = export_finetune_jsonl(args.input, args.output, fmt=args.format)
    print(
        f"Exported {stats['exported']} example(s) to {args.output} "
        f"({stats['skipped']} skipped, {stats['total']} total records)"
    )
    return 0 if stats["exported"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
