#!/usr/bin/env python3
"""Upload README model card so HF Inference Endpoints enables TGI/vLLM."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish HF model card metadata")
    parser.add_argument(
        "--repo-id",
        default="sreenuti/realpage-message-agent-v1",
        help="Hub model repo id",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=ROOT / "models" / "realpage-message-agent-v1" / "README.md",
    )
    args = parser.parse_args(argv)

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("Set HF_TOKEN in .env", file=sys.stderr)
        return 1

    if not args.readme.exists():
        print(f"Missing {args.readme}", file=sys.stderr)
        return 1

    from huggingface_hub import HfApi, model_info

    api = HfApi(token=token)
    readme_text = args.readme.read_text(encoding="utf-8")

    print(f"Uploading README to {args.repo_id} ...")
    api.upload_file(
        path_or_fileobj=readme_text.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Add model card for TGI/vLLM Inference Endpoints",
    )

    info = model_info(args.repo_id, token=token)
    print(f"pipeline_tag: {getattr(info, 'pipeline_tag', None)}")
    print(f"library_name: {getattr(info, 'library_name', None)}")
    print(f"tags: {info.tags}")
    print(f"\nModel page: https://huggingface.co/{args.repo_id}")
    print("Refresh the Create Endpoint page; TGI / vLLM should unlock.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
