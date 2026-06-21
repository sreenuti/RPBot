"""Agent pipeline runner with full trace capture."""

from __future__ import annotations

import os
import time
import uuid
from collections import Counter
from datetime import datetime, timezone

from src.evaluator import check_thresholds, evaluate
from src.llm_client import LLMClient, LLMError
from src.llm_judge import run_llm_judge
from src.output_parser import OutputParseError, parse_agent_output
from src.output_sanitizer import sanitize_output
from src.prompt_builder import build_prompt, build_retry_prompt, record_to_context
from src.schemas import AgentOutput, InputRecord
from src.trace import RecordTrace, RunSummary, RunTrace, TraceStep, StepPhase, StepStatus
from src.validator import validate

MAX_VALIDATION_RETRIES = 2


def _step(
    steps: list[TraceStep],
    *,
    phase: StepPhase,
    title: str,
    status: StepStatus = "info",
    elapsed_ms: int = 0,
    message: str | None = None,
    data: dict | None = None,
) -> TraceStep:
    step = TraceStep(
        step_id=f"{phase}-{len(steps) + 1}",
        phase=phase,
        title=title,
        status=status,
        elapsed_ms=elapsed_ms,
        message=message,
        data=data or {},
    )
    steps.append(step)
    return step


def process_record(
    record: InputRecord,
    llm: LLMClient,
    *,
    judge_llm: LLMClient | None = None,
    verbose: bool = False,
    capture_trace: bool = False,
) -> tuple[AgentOutput, RecordTrace | None]:
    """Process a single record; optionally return a detailed execution trace."""
    trace_steps: list[TraceStep] = []
    run_start = time.perf_counter()

    if capture_trace:
        _step(
            trace_steps,
            phase="ingest",
            title="Load input record",
            status="success",
            message=f"Loaded task {record.task_id}",
            data={
                "task_id": record.task_id,
                "persona": record.persona,
                "lifecycle_stage": record.lifecycle_stage,
                "channel_preferences": record.channel_preferences,
                "consent": record.consent.model_dump(),
                "input_context": record_to_context(record),
            },
        )

    prompt = build_prompt(record)
    validation_errors: list[str] = []
    output: AgentOutput | None = None
    llm_calls = 0
    llm_latency_ms = 0

    for attempt in range(MAX_VALIDATION_RETRIES + 1):
        attempt_start = time.perf_counter()
        current_prompt = prompt if attempt == 0 else build_retry_prompt(record, validation_errors)

        if capture_trace:
            phase = "prompt" if attempt == 0 else "retry"
            _step(
                trace_steps,
                phase=phase,
                title="Build agent prompt" if attempt == 0 else f"Retry prompt (attempt {attempt + 1})",
                status="warning" if attempt > 0 else "info",
                message=(
                    "Composed autonomous decision prompt from input record."
                    if attempt == 0
                    else f"Previous attempt failed validation: {'; '.join(validation_errors)}"
                ),
                data={
                    "attempt": attempt + 1,
                    "prompt_preview": current_prompt[:1200] + ("…" if len(current_prompt) > 1200 else ""),
                    "prompt_length": len(current_prompt),
                    "full_prompt": current_prompt,
                },
            )

        llm_start = time.perf_counter()
        try:
            llm_result = llm.generate(current_prompt, record=record)
        except LLMError as exc:
            if capture_trace:
                _step(
                    trace_steps,
                    phase="error",
                    title="LLM invocation failed",
                    status="error",
                    elapsed_ms=int((time.perf_counter() - llm_start) * 1000),
                    message=str(exc),
                )
            raise

        llm_calls += 1
        llm_elapsed = int((time.perf_counter() - llm_start) * 1000)
        llm_latency_ms += llm_elapsed

        if capture_trace:
            _step(
                trace_steps,
                phase="llm",
                title=f"LLM decision (call #{llm_calls})",
                status="success",
                elapsed_ms=llm_elapsed,
                message=(
                    "Mock test double returned a simulated autonomous decision."
                    if llm.mock
                    else f"{llm.provider} model generated structured JSON decision."
                ),
                data={
                    "provider": llm.provider,
                    "mock": llm.mock,
                    "model": llm.model_name,
                    "attempt": attempt + 1,
                    "raw_response": llm_result,
                    "chain_of_thought": llm_result.get("reasoning"),
                },
            )

        parse_start = time.perf_counter()
        try:
            output = parse_agent_output(record.task_id, llm_result)
            parse_status = "success"
            parse_message = "Parsed LLM JSON into AgentOutput schema."
            parse_error = None
        except OutputParseError as exc:
            validation_errors = [str(exc)]
            parse_status = "error"
            parse_message = str(exc)
            parse_error = str(exc)
            output = None

        if capture_trace:
            _step(
                trace_steps,
                phase="parse",
                title="Parse structured output",
                status=parse_status,
                elapsed_ms=int((time.perf_counter() - parse_start) * 1000),
                message=parse_message,
                data={
                    "parsed": output.model_dump() if output else None,
                    "error": parse_error,
                },
            )

        if output is None:
            if attempt == MAX_VALIDATION_RETRIES:
                raise LLMError(f"Failed to parse LLM output for {record.task_id}: {parse_error}")
            continue

        output = sanitize_output(output, record)

        validate_start = time.perf_counter()
        validation_errors = validate(output, record)
        validate_elapsed = int((time.perf_counter() - validate_start) * 1000)

        if capture_trace:
            _step(
                trace_steps,
                phase="validate",
                title="Policy & schema validation",
                status="success" if not validation_errors else "warning",
                elapsed_ms=validate_elapsed,
                message=(
                    "All consent, safety, and schema checks passed."
                    if not validation_errors
                    else f"{len(validation_errors)} issue(s) found."
                ),
                data={"errors": validation_errors, "passed": not validation_errors},
            )

        if not validation_errors:
            break

        if verbose:
            print(f"[warn] {record.task_id} attempt {attempt + 1}: {'; '.join(validation_errors)}")

        if attempt == MAX_VALIDATION_RETRIES:
            break

        if capture_trace:
            _step(
                trace_steps,
                phase="retry",
                title="Schedule validation retry",
                status="warning",
                elapsed_ms=int((time.perf_counter() - attempt_start) * 1000),
                message="Sending validation feedback back to the LLM for correction.",
                data={"errors": validation_errors},
            )

    if output is None:
        raise LLMError(f"No output produced for {record.task_id}")

    total_latency_ms = int((time.perf_counter() - run_start) * 1000)
    output.quality = evaluate(output, record, latency_ms=llm_latency_ms or total_latency_ms)

    threshold_result = check_thresholds(output.quality, record)
    if capture_trace:
        _step(
            trace_steps,
            phase="evaluate",
            title="Quality evaluation",
            status="success",
            message="Computed personalization score and safety metrics.",
            data={
                "personalization_score": output.quality.personalization_score,
                "safety_violations": output.quality.safety_violations,
                "latency_ms": output.quality.latency_ms,
            },
        )

    if judge_llm:
        judge_start = time.perf_counter()
        try:
            output.llm_judge = run_llm_judge(record, output, judge_llm)
            judge_elapsed = int((time.perf_counter() - judge_start) * 1000)
            if capture_trace:
                _step(
                    trace_steps,
                    phase="judge",
                    title="Gemini judge",
                    status="success" if output.llm_judge.passed else "warning",
                    elapsed_ms=judge_elapsed,
                    message=output.llm_judge.reasoning or "Judge evaluation complete.",
                    data=output.llm_judge.model_dump(),
                )
        except LLMError as exc:
            if capture_trace:
                _step(
                    trace_steps,
                    phase="judge",
                    title="Gemini judge",
                    status="error",
                    elapsed_ms=int((time.perf_counter() - judge_start) * 1000),
                    message=str(exc),
                )
            if verbose:
                print(f"[warn] {record.task_id} judge failed: {exc}")

    if capture_trace:
        _step(
            trace_steps,
            phase="threshold",
            title="Threshold check",
            status="success" if threshold_result.passed else "warning",
            message=(
                "All record thresholds met."
                if threshold_result.passed
                else f"Threshold breaches: {'; '.join(threshold_result.failures)}"
            ),
            data={
                "passed": threshold_result.passed,
                "failures": threshold_result.failures,
                "thresholds": record.thresholds.model_dump(),
            },
        )
        _step(
            trace_steps,
            phase="complete",
            title="Agent run complete",
            status="success" if threshold_result.passed and not validation_errors else "warning",
            elapsed_ms=total_latency_ms,
            message=output.reasoning,
            data={
                "should_send": output.should_send,
                "channel": output.next_message.channel,
                "send_at": output.next_message.send_at,
                "next_action": output.next_action.model_dump(),
            },
        )

    if validation_errors and verbose:
        print(
            f"[warn] {record.task_id} unresolved after retries: "
            f"{'; '.join(validation_errors)}"
        )
    if not threshold_result.passed and verbose:
        print(f"[warn] {record.task_id} thresholds: {'; '.join(threshold_result.failures)}")

    record_trace = None
    if capture_trace:
        record_trace = RecordTrace(
            task_id=record.task_id,
            input_record=record.model_dump(),
            steps=trace_steps,
            output=output,
            latency_ms=total_latency_ms,
        )

    return output, record_trace


def _build_summary(outputs: list[AgentOutput], records: list[InputRecord]) -> RunSummary:
    total = len(outputs)
    sent = sum(1 for o in outputs if o.should_send)
    channels = Counter(
        (o.next_message.channel or "none") if o.should_send else "suppressed"
        for o in outputs
    )
    scores = [o.quality.personalization_score for o in outputs]
    latencies = [o.quality.latency_ms or 0 for o in outputs]
    max_safety = max((o.quality.safety_violations for o in outputs), default=0)
    threshold_passed = 0
    judge_scores: list[float] = []
    judge_passed = 0
    judge_count = 0
    for output, record in zip(outputs, records):
        if check_thresholds(output.quality, record).passed:
            threshold_passed += 1
        if output.llm_judge is not None:
            judge_count += 1
            judge_scores.append(output.llm_judge.overall_score)
            if output.llm_judge.passed:
                judge_passed += 1

    return RunSummary(
        total_records=total,
        sent=sent,
        suppressed=total - sent,
        channel_distribution=dict(channels),
        average_personalization_score=round(sum(scores) / len(scores), 2) if scores else 0.0,
        max_safety_violations=max_safety,
        average_latency_ms=round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        threshold_pass_rate=round(threshold_passed / total, 2) if total else 0.0,
        judge_pass_rate=round(judge_passed / judge_count, 2) if judge_count else None,
        average_judge_score=round(sum(judge_scores) / len(judge_scores), 2)
        if judge_scores
        else None,
    )


def run_batch(
    records: list[InputRecord],
    llm: LLMClient,
    *,
    judge_llm: LLMClient | None = None,
    verbose: bool = False,
    capture_trace: bool = False,
) -> tuple[list[AgentOutput], RunTrace | None]:
    """Process multiple records and optionally capture a full run trace."""
    run_start = time.perf_counter()
    outputs: list[AgentOutput] = []
    record_traces: list[RecordTrace] = []

    if (
        not llm.mock
        and os.getenv("LLM_WARMUP", "false").lower() in ("1", "true", "yes")
    ):
        llm.warmup()

    for record in records:
        output, trace = process_record(
            record,
            llm,
            judge_llm=judge_llm,
            verbose=verbose,
            capture_trace=capture_trace,
        )
        outputs.append(output)
        if trace:
            record_traces.append(trace)

    run_trace = None
    if capture_trace:
        total_ms = int((time.perf_counter() - run_start) * 1000)
        run_trace = RunTrace(
            run_id=str(uuid.uuid4()),
            mock=llm.mock,
            provider=llm.provider,
            model=llm.model_name,
            started_at=datetime.now(timezone.utc).isoformat(),
            total_latency_ms=total_ms,
            summary=_build_summary(outputs, records),
            records=record_traces,
        )

    return outputs, run_trace
