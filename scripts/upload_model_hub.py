#!/usr/bin/env python3
"""Upload a merged model directory to the Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _verify_model_dir(model_dir: Path) -> list[str]:
    import importlib.util

    path = ROOT / "scripts" / "verify_model_export.py"
    spec = importlib.util.spec_from_file_location("verify_model_export", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.verify_model_dir(model_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Push merged RealPage model to Hugging Face Hub"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Local merged model folder",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hub repo id, e.g. yourusername/realpage-message-agent-v1",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/update as a private repo",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload RealPage merged LoRA model",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip local weight file verification",
    )
    args = parser.parse_args(argv)

    model_dir = args.model_dir.resolve()
    if not args.skip_verify:
        errors = _verify_model_dir(model_dir)
        if errors:
            print("Model verification failed:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            return 1

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print(
            "Set HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) with a Write token from "
            "https://huggingface.co/settings/tokens\n"
            "  PowerShell: $env:HF_TOKEN = \"hf_...\"\n"
            "  Or add HF_TOKEN=hf_... to .env",
            file=sys.stderr,
        )
        return 1

    if not token.startswith("hf_"):
        print(
            "WARNING: HF tokens usually start with hf_. Check you copied the full token.",
            file=sys.stderr,
        )

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError as exc:
        print(
            "Install huggingface_hub: pip install huggingface_hub",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    api = HfApi(token=token)
    try:
        create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True,
            token=token,
        )
    except Exception as exc:
        if "401" in str(exc) or "Unauthorized" in str(exc):
            print(
                "401 Unauthorized — your HF token was rejected.\n"
                "  1. Create a new token: https://huggingface.co/settings/tokens\n"
                "  2. Type: Fine-grained or Classic with **Write** access\n"
                "  3. Ensure the token belongs to user 'sreenuti' (matches --repo-id)\n"
                "  4. Set: $env:HF_TOKEN = \"hf_...\"  then re-run upload",
                file=sys.stderr,
            )
        raise SystemExit(1) from exc

    print(f"Uploading {model_dir} -> hf.co/{args.repo_id} ...")
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.commit_message,
    )

    print(f"\nDone: https://huggingface.co/{args.repo_id}")
    print("\nNext: create an Inference Endpoint (TGI) — see docs/HOSTING.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
