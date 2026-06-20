#!/usr/bin/env python3
"""Convert generated training rows into LoRA chat JSONL format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pydantic import BaseModel, ValidationError  # noqa: E402

from scripts.generate_training_data import (  # noqa: E402
    GeneratedTrainingRow,
    validate_generated_row,
)

SYSTEM_PROMPT = (
    "You are a RealPage communication decision agent. "
    "Return only valid JSON matching the expected schema."
)


class LoraTrainingRow(BaseModel):
    messages: list[dict[str, str]]


def load_generated_jsonl(path: Path) -> list[GeneratedTrainingRow]:
    rows: list[GeneratedTrainingRow] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                rows.append(GeneratedTrainingRow.model_validate(payload))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(f"{path}:{line_no}: invalid row: {exc}") from exc
    return rows


def to_lora_row(row: GeneratedTrainingRow) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(row.input, ensure_ascii=False)},
            {"role": "assistant", "content": json.dumps(row.expected, ensure_ascii=False)},
        ]
    }


def export_lora_jsonl(input_path: Path, output_path: Path) -> dict[str, int]:
    rows = load_generated_jsonl(input_path)
    errors: list[str] = []
    exported = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(rows):
            row_errors = validate_generated_row(row)
            if row_errors:
                for err in row_errors:
                    errors.append(f"row {idx}: {err}")
                continue
            lora_row = to_lora_row(row)
            LoraTrainingRow.model_validate(lora_row)
            handle.write(json.dumps(lora_row, ensure_ascii=False))
            handle.write("\n")
            exported += 1

    if errors:
        raise ValueError("Validation failed:\n" + "\n".join(errors[:20]))

    return {"total": len(rows), "exported": exported, "skipped": len(rows) - exported}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export generated rows to LoRA chat JSONL")
    parser.add_argument("--input", required=True, help="Generated JSONL with input/expected rows")
    parser.add_argument("--output", required=True, help="Output LoRA training JSONL path")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1

    try:
        stats = export_lora_jsonl(input_path, output_path)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(
        f"Exported {stats['exported']} example(s) to {output_path} "
        f"({stats['skipped']} skipped, {stats['total']} total records)"
    )
    return 0 if stats["exported"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
