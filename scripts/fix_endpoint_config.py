#!/usr/bin/env python3
"""Patch model config on Hub for T4 Inference Endpoints (reduce context / vLLM OOM)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = ROOT / "models" / "realpage-message-agent-v1"
DEFAULT_MAX_LEN = 4096


def patch_config(config_path: Path, max_len: int) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    old = config.get("max_position_embeddings")
    config["max_position_embeddings"] = max_len
    config["use_cache"] = True
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"max_position_embeddings: {old} -> {max_len}")
    return config


def upload_files(repo_id: str, model_dir: Path, token: str, files: list[str]) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    for name in files:
        path = model_dir / name
        api.upload_file(
            path_or_fileobj=path.read_bytes(),
            path_in_repo=name,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Patch {name} for T4/vLLM deployment (max_len={DEFAULT_MAX_LEN})",
        )
        print(f"Uploaded {name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Patch config for HF endpoint deployment")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--repo-id", default="sreenuti/realpage-message-agent-v1")
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args(argv)

    model_dir = args.model_dir.resolve()
    config_path = model_dir / "config.json"
    if not config_path.exists():
        print(f"Missing {config_path}", file=sys.stderr)
        return 1

    patch_config(config_path, args.max_len)

    if args.upload:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if not token:
            print("Set HF_TOKEN to upload", file=sys.stderr)
            return 1
        upload_files(args.repo_id, model_dir, token, ["config.json"])

    print("\nRecreate or restart the endpoint with Environment Variables:")
    print("  MAX_MODEL_LEN=4096")
    print("  GPU_MEMORY_UTILIZATION=0.75")
    print("Or switch Inference Engine to Text Generation Inference (TGI).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
