#!/usr/bin/env python3
"""Zip files needed for Colab training upload."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Package Colab upload zip")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "colab_upload.zip",
        help="Output zip path",
    )
    args = parser.parse_args()

    files = [
        ROOT / "data" / "finetune" / "train.jsonl",
        ROOT / "notebooks" / "train_lora_colab.ipynb",
    ]
    missing = [path for path in files if not path.exists()]
    if missing:
        raise SystemExit(f"Missing: {', '.join(str(p) for p in missing)}")

    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            arcname = path.relative_to(ROOT).as_posix()
            zf.write(path, arcname=arcname)

    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"Wrote {args.output} ({size_mb:.1f} MB)")
    print("Upload train.jsonl via Colab notebook cell, or unzip on Drive.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
