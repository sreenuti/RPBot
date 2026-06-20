"""JSONL file loader with graceful error handling."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from src.schemas import InputRecord


class LoaderError(Exception):
    """Raised when JSONL loading or validation fails."""


def load_jsonl(path: str | Path) -> list[InputRecord]:
    """Read a JSONL file and return validated input records."""
    file_path = Path(path)
    if not file_path.exists():
        raise LoaderError(f"Input file not found: {file_path}")

    records: list[InputRecord] = []
    with file_path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LoaderError(
                    f"Line {line_number}: invalid JSON: {exc.msg}"
                ) from exc
            try:
                records.append(InputRecord.model_validate(payload))
            except ValidationError as exc:
                raise LoaderError(
                    f"Line {line_number}: schema error: {exc}"
                ) from exc
    return records
