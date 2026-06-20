#!/usr/bin/env python3
"""Fix tokenizer_config.json for vLLM/TGI (transformers 4.x+ on Inference Endpoints)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = ROOT / "models" / "realpage-message-agent-v1"
DEFAULT_BASE = "Qwen/Qwen2.5-1.5B-Instruct"


def fetch_base_tokenizer_config(base_model: str) -> dict:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(base_model, "tokenizer_config.json")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fix_local_model_dir(model_dir: Path, base_model: str) -> None:
    cfg = fetch_base_tokenizer_config(base_model)
    out_path = model_dir / "tokenizer_config.json"
    out_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


def upload_to_hub(repo_id: str, model_dir: Path, token: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    readme = model_dir / "tokenizer_config.json"
    api.upload_file(
        path_or_fileobj=readme.read_bytes(),
        path_in_repo="tokenizer_config.json",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Fix tokenizer_config for vLLM Inference Endpoints",
    )
    print(f"Uploaded tokenizer_config.json to {repo_id}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fix and publish tokenizer_config.json")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--base-model", default=DEFAULT_BASE)
    parser.add_argument("--repo-id", default="sreenuti/realpage-message-agent-v1")
    parser.add_argument("--upload", action="store_true", help="Upload fix to Hugging Face Hub")
    args = parser.parse_args(argv)

    model_dir = args.model_dir.resolve()
    if not model_dir.is_dir():
        print(f"Missing {model_dir}", file=sys.stderr)
        return 1

    fix_local_model_dir(model_dir, args.base_model)

    if args.upload:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if not token:
            print("Set HF_TOKEN to upload", file=sys.stderr)
            return 1
        upload_to_hub(args.repo_id, model_dir, token)

    print("Done. Restart or recreate the Inference Endpoint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
