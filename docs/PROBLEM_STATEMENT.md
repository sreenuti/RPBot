# Problem Statement

Build a context-aware message-sending bot that learns what to do only from input data.

You are given a JSONL file (see `data/sample.jsonl`) where each line is a test case:

- **Input:** user profile, preferences, context, constraints
- **Expected:** what message should be sent (or not sent), via which channel, and why

Your job is to build an autonomous agent that:

- Reads the input record
- Decides if it should communicate
- Decides how to communicate
- Decides what to say
- Produces output that semantically matches the expected result

## Solution Approach (RPBot)

This repository implements the above with:

1. **LLM autonomy** — all communication decisions inferred from input fields
2. **Policy validation** — consent, Fair Housing, PII, and opt-out enforced after generation
3. **Quality evaluation** — personalization scoring and threshold checks (metrics only)
4. **Hold-out safety** — `expected` labels never sent to the LLM at runtime
5. **Demo UI** — interactive pipeline trace for stakeholder review

See [ARCHITECTURE.md](./ARCHITECTURE.md) for full system design.
