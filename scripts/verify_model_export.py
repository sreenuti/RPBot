#!/usr/bin/env python3
"""Verify a merged model directory is complete before Hub upload or local serve."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt")


def verify_model_dir(model_dir: Path) -> list[str]:
    errors: list[str] = []
    if not model_dir.is_dir():
        return [f"Not a directory: {model_dir}"]

    if not (model_dir / "config.json").exists():
        errors.append("Missing config.json")

    weight_files = [
        path
        for path in model_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in WEIGHT_SUFFIXES
    ]
    if not weight_files:
        errors.append(
            "No weight files (.safetensors / .bin) found. "
            "Re-run Colab training with merge=True and re-download the zip."
        )
    else:
        total_bytes = sum(path.stat().st_size for path in weight_files)
        total_gb = total_bytes / (1024**3)
        if total_gb < 0.5:
            errors.append(
                f"Weight files total only {total_gb:.2f} GB — export may be incomplete."
            )

    meta_path = model_dir / "realpage_model.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            print(f"  base_model: {meta.get('base_model')}")
            print(f"  format: {meta.get('format')}")
        except json.JSONDecodeError:
            errors.append("realpage_model.json is invalid JSON")
    else:
        print("  (no realpage_model.json — optional metadata file)")

    tokenizer_ok = (model_dir / "tokenizer.json").exists() or (
        model_dir / "tokenizer_config.json"
    ).exists()
    if not tokenizer_ok:
        errors.append("Missing tokenizer files (tokenizer.json or tokenizer_config.json)")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify merged model export")
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to merged model folder (e.g. models/realpage-message-agent-v1)",
    )
    args = parser.parse_args(argv)

    model_dir = args.model_dir.resolve()
    print(f"Checking {model_dir}\n")

    errors = verify_model_dir(model_dir)
    if errors:
        print("FAILED:")
        for error in errors:
            print(f"  - {error}")
        return 1

    weight_files = [
        p for p in model_dir.rglob("*") if p.suffix.lower() in WEIGHT_SUFFIXES
    ]
    total_gb = sum(p.stat().st_size for p in weight_files) / (1024**3)
    print("OK — model export looks complete.")
    print(f"  Weight files: {len(weight_files)} ({total_gb:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
