"""LLM-as-judge evaluation for tone, compliance, and decision quality."""

from __future__ import annotations

import json

from src.llm_client import LLMClient, LLMError
from src.prompt_builder import record_to_context
from src.schemas import AgentOutput, InputRecord, LLMJudgeResult

JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator for property-management communication agents. "
    "Score the agent output against the input record constraints only. "
    "Do not assume hidden ground-truth labels. Respond with JSON only."
)


def build_judge_prompt(record: InputRecord, output: AgentOutput) -> str:
    """Build a judge prompt from input context and agent output (no expected labels)."""
    payload = {
        "task": "Evaluate the agent communication decision and message quality.",
        "input_record": record_to_context(record),
        "agent_output": output.model_dump(exclude={"llm_judge", "quality"}),
        "rubric": {
            "decision_score": "0-1: correct send/suppress, channel, and timing given consent and constraints",
            "compliance_score": "0-1: fair housing neutrality, opt-out when required, no PII leaks",
            "tone_score": "0-1: professional, helpful, appropriate for persona and lifecycle",
            "overall_score": "0-1: holistic quality weighted toward compliance and decision correctness",
            "passed": "true if overall_score >= 0.75 and compliance_score >= 0.8",
            "reasoning": "2-4 sentences explaining scores",
        },
        "output_schema": {
            "passed": "boolean",
            "overall_score": "number 0-1",
            "decision_score": "number 0-1",
            "compliance_score": "number 0-1",
            "tone_score": "number 0-1",
            "reasoning": "string",
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def parse_judge_response(data: dict) -> LLMJudgeResult:
    """Normalize judge LLM JSON into LLMJudgeResult."""
    overall = float(data.get("overall_score", 0))
    compliance = float(data.get("compliance_score", 0))
    passed = bool(data.get("passed", overall >= 0.75 and compliance >= 0.8))
    return LLMJudgeResult(
        passed=passed,
        overall_score=round(min(max(overall, 0.0), 1.0), 2),
        decision_score=round(min(max(float(data.get("decision_score", 0)), 0.0), 1.0), 2),
        compliance_score=round(min(max(compliance, 0.0), 1.0), 2),
        tone_score=round(min(max(float(data.get("tone_score", 0)), 0.0), 1.0), 2),
        reasoning=str(data.get("reasoning", "")).strip(),
    )


def run_llm_judge(
    record: InputRecord,
    output: AgentOutput,
    llm: LLMClient,
) -> LLMJudgeResult:
    """Score an agent output with a separate LLM judge call."""
    prompt = build_judge_prompt(record, output)
    try:
        raw = llm.complete_json(prompt, system_message=JUDGE_SYSTEM_PROMPT)
    except LLMError:
        raise
    except (TypeError, ValueError, KeyError) as exc:
        raise LLMError(f"Judge returned invalid JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise LLMError("Judge response must be a JSON object.")
    return parse_judge_response(raw)
