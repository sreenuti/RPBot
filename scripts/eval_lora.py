#!/usr/bin/env python3
"""Evaluate a local fine-tuned model against labeled expected outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.finetune_export import build_training_target  # noqa: E402
from src.loader import LoaderError, load_jsonl  # noqa: E402
from src.output_parser import OutputParseError  # noqa: E402

DEFAULT_INPUT = ROOT / "data" / "test_cases.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs" / "lora_eval.jsonl"

HEAVY_SCRIPT_MARKERS = ("train_lora.py",)


def _warn_if_heavy_python_jobs_running() -> None:
    """Warn when training/serving is already using GPU RAM alongside eval."""
    try:
        import psutil
    except ImportError:
        return

    conflicts: list[str] = []
    current = psutil.Process().pid
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            if proc.info["pid"] == current:
                continue
            cmdline = proc.info.get("cmdline") or []
            text = " ".join(cmdline)
            if "python" not in text.lower():
                continue
            for marker in HEAVY_SCRIPT_MARKERS:
                if marker in text:
                    conflicts.append(f"{marker} (pid {proc.info['pid']})")
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if conflicts:
        print(
            "WARNING: Other heavy GPU jobs are running:\n  "
            + "\n  ".join(conflicts)
            + "\nStop training before serve+eval on a 6GB GPU, or eval with --mock.",
            file=sys.stderr,
        )


def _check_local_server(base_url: str) -> None:
    """Fail fast if local provider is configured but the server is not reachable."""
    import urllib.error
    import urllib.request

    health_url = base_url.rstrip("/").removesuffix("/v1") + "/health"
    try:
        with urllib.request.urlopen(health_url, timeout=5) as response:
            if response.status != 200:
                raise RuntimeError(f"unexpected status {response.status}")
    except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
        raise SystemExit(
            "Local model server is not reachable at "
            f"{health_url}. Start it first:\n"
            "  python scripts/serve_local_model.py --model-dir models/<your-model>\n"
            f"Details: {exc}"
        ) from exc


def _compare_field(name: str, actual: object, expected: object) -> str | None:
    if actual == expected:
        return None
    return f"{name}: expected {expected!r}, got {actual!r}"


def _compare_output(actual: dict, expected: dict) -> list[str]:
    mismatches: list[str] = []

    for field in ("should_send",):
        mismatch = _compare_field(field, actual.get(field), expected.get(field))
        if mismatch:
            mismatches.append(mismatch)

    actual_message = actual.get("next_message") or {}
    expected_message = expected.get("next_message") or {}
    for field in ("channel", "send_at", "subject"):
        mismatch = _compare_field(
            f"next_message.{field}",
            actual_message.get(field),
            expected_message.get(field),
        )
        if mismatch:
            mismatches.append(mismatch)

    actual_action = actual.get("next_action") or {}
    expected_action = expected.get("next_action") or {}
    mismatch = _compare_field(
        "next_action.type",
        actual_action.get("type"),
        expected_action.get("type"),
    )
    if mismatch:
        mismatches.append(mismatch)

    return mismatches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the agent on labeled JSONL and compare to expected outputs"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mock", action="store_true", help="Use mock LLM instead of API")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    try:
        records = load_jsonl(args.input)
    except LoaderError as exc:
        print(f"Error loading input: {exc}", file=sys.stderr)
        return 1

    labeled = [record for record in records if record.expected]
    if not labeled:
        print("No labeled records with expected outputs found.", file=sys.stderr)
        return 1

    import os
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    provider = os.getenv("LLM_PROVIDER", "openai")
    if not args.mock and provider != "local":
        print(
            "WARNING: LLM_PROVIDER is not 'local'. "
            "Set LLM_PROVIDER=local in .env or pass --mock.",
            file=sys.stderr,
        )

    if not args.mock:
        _warn_if_heavy_python_jobs_running()
        if provider == "local":
            base_url = os.getenv("LOCAL_BASE_URL") or os.getenv("OPENAI_BASE_URL") or ""
            if base_url:
                _check_local_server(base_url)
            else:
                print(
                    "ERROR: LLM_PROVIDER=local but LOCAL_BASE_URL is not set.",
                    file=sys.stderr,
                )
                return 1

    from src.agent_runner import process_record
    from src.exporter import export_jsonl
    from src.llm_client import LLMClient, LLMError

    llm = LLMClient(mock=args.mock)
    outputs = []
    run_errors: list[dict] = []

    print(f"Running agent on {len(records)} record(s)...")
    for index, record in enumerate(records, start=1):
        print(f"[{index}/{len(records)}] {record.task_id}", flush=True)
        try:
            output, _ = process_record(record, llm, verbose=args.verbose)
            outputs.append(output)
        except LLMError as exc:
            run_errors.append({"task_id": record.task_id, "error": str(exc)})
            print(f"  error: {exc}", file=sys.stderr)

    export_jsonl(outputs, args.output)
    print(f"Wrote {len(outputs)} output(s) to {args.output}")
    if run_errors:
        print(f"{len(run_errors)} record(s) failed during inference.", file=sys.stderr)

    outputs_by_task: dict[str, dict] = {}
    with args.output.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                outputs_by_task[row["task_id"]] = row

    passed = 0
    failed = 0
    invalid = 0
    results: list[dict] = []

    for record in labeled:
        actual_row = outputs_by_task.get(record.task_id)
        if actual_row is None:
            invalid += 1
            inference_error = next(
                (item["error"] for item in run_errors if item["task_id"] == record.task_id),
                None,
            )
            mismatches = [inference_error or "no output produced"]
            results.append(
                {
                    "task_id": record.task_id,
                    "status": "missing_output",
                    "mismatches": mismatches,
                }
            )
            continue

        try:
            expected = build_training_target(record)
        except (ValueError, OutputParseError) as exc:
            invalid += 1
            results.append(
                {
                    "task_id": record.task_id,
                    "status": "bad_expected",
                    "mismatches": [str(exc)],
                }
            )
            continue

        actual = {
            "should_send": actual_row.get("should_send"),
            "next_message": actual_row.get("next_message"),
            "next_action": actual_row.get("next_action"),
        }
        mismatches = _compare_output(actual, expected)
        if mismatches:
            failed += 1
            status = "fail"
        else:
            passed += 1
            status = "pass"

        results.append(
            {
                "task_id": record.task_id,
                "status": status,
                "mismatches": mismatches,
            }
        )

    summary_path = args.output.with_suffix(".summary.json")
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "total": len(labeled),
        "passed": passed,
        "failed": failed,
        "invalid": invalid,
        "run_errors": run_errors,
        "results": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nEvaluation summary")
    print(f"  Total labeled records: {len(labeled)}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Invalid/missing: {invalid}")
    print(f"  Summary written to {summary_path}")

    if failed or invalid:
        print("\nMismatches:")
        for item in results:
            if item["status"] != "pass":
                print(f"  - {item['task_id']}: {', '.join(item['mismatches'])}")

    return 0 if failed == 0 and invalid == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
