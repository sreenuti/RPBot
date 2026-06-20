# Testing Guide

## Quick Start

```bash
cd realpage-message-agent
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
pytest
```

All tests run **without API keys** using mock mode and unit tests.

## Test Structure

```
tests/
├── conftest.py              # Adds project root to sys.path
├── test_loader.py           # JSONL loading and error handling
├── test_agent.py            # Output parsing and mock LLM decisions
├── test_validator.py        # Consent, safety, schema rules
├── test_evaluator.py        # Personalization scoring and thresholds
├── test_prompt_builder.py   # Prompt construction and hold-out safety
├── test_ui_api.py           # FastAPI endpoint integration
└── test_mock_run.py         # End-to-end CLI and subprocess runs
```

## Test Categories

### Unit Tests

| File | What it verifies |
|------|------------------|
| `test_loader.py` | Valid JSONL parsing, malformed JSON, schema errors, empty line skipping |
| `test_agent.py` | `parse_agent_output()`, mock channel selection, opt-out suppression |
| `test_validator.py` | Email subject rules, SMS null subject, opt-out, discrimination blocklist, consent |
| `test_evaluator.py` | Personalization weights, threshold pass/fail, suppressed sends score 0 |
| `test_prompt_builder.py` | `expected` excluded from prompts, retry prompt includes errors |

### Integration Tests

| File | What it verifies |
|------|------------------|
| `test_mock_run.py` | CLI `run.py --mock` produces valid JSONL; subprocess invocation |
| `test_ui_api.py` | Health, sample, run endpoints via FastAPI TestClient |

## Running Specific Tests

```bash
# Single file
pytest tests/test_validator.py -v

# Single test
pytest tests/test_validator.py::test_opt_out_required -v

# With coverage (requires pytest-cov: pip install pytest-cov)
pytest --cov=src --cov-report=term-missing
```

## Mock End-to-End Run

```bash
python run.py --input data/sample.jsonl --output outputs/outputs.jsonl --mock --verbose
```

Expected: 2 records processed, SMS + email channels, valid JSONL output.

## What Tests Do NOT Assert

- Exact message text matching `expected` gold labels (LLM autonomy)
- Real OpenAI/Gemini API responses (requires live integration tests with keys)
- Non-deterministic timing beyond threshold structure

## CI Recommendation

```yaml
# Example GitHub Actions step
- name: Run tests
  run: |
    pip install -r requirements.txt
    pytest tests/ -v
```

No secrets required for the default test suite.

## Adding New Tests

1. Place tests in `tests/test_<module>.py`
2. Use fixtures from `data/sample.jsonl` via `load_jsonl()`
3. For validator tests, build synthetic `AgentOutput` with `_base_output()` pattern
4. Keep tests independent—no shared mutable state
