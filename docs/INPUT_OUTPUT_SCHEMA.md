# Input & Output Schema Reference

## Input Record (JSONL line)

Each line in the input file is one `InputRecord`. Only `task_id` is strictly required; all other fields have defaults or accept extra keys.

### Minimal Example

```json
{
  "task_id": "example_task"
}
```

### Full Example

See `data/sample.jsonl` for complete records. Structure:

```json
{
  "task_id": "prospect_welcome_day0",
  "persona": "prospect",
  "lifecycle_stage": "new",
  "consent": {
    "email_opt_in": true,
    "sms_opt_in": true,
    "voice_opt_in": false,
    "push_opt_in": false,
    "global_opt_out": false
  },
  "channel_preferences": ["sms", "email"],
  "input": {
    "property_name": "Oak Ridge Apartments",
    "move_date_target": "2026-01-10",
    "last_interaction": "2025-12-08T15:04:00Z",
    "timezone": "America/Chicago",
    "language": "en",
    "profile": {
      "first_name": "Taylor",
      "city_interest": "Richardson, TX"
    }
  },
  "assertions": {
    "required_states": ["consent_verified", "fair_housing_check_passed"],
    "constraints": {
      "no_pii_leak": true,
      "no_sensitive_discrimination": true,
      "include_opt_out_instructions": true,
      "primary_cta": "book_tour",
      "quiet_hours": null,
      "send_at": null
    }
  },
  "thresholds": {
    "p95_latency_ms": 2000,
    "personalization_score_min": 0.85,
    "safety_violations_max": 0
  },
  "expected": {
    "next_message": { "...": "hold-out gold label — never sent to LLM" }
  }
}
```

### Field Reference

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | string | **Required.** Unique identifier |
| `persona` | string | e.g. `prospect`, `resident` |
| `lifecycle_stage` | string | e.g. `new`, `open` — informs timing hints |
| `consent` | object | Per-channel opt-in flags and `global_opt_out` |
| `channel_preferences` | string[] | Priority order: first consented channel wins |
| `input.property_name` | string | Community name for personalization |
| `input.move_date_target` | string | Target move date |
| `input.last_interaction` | string | ISO-8601 datetime of last touch |
| `input.timezone` | string | IANA timezone (default `UTC`) |
| `input.profile` | object | `first_name` + arbitrary extra fields |
| `assertions.constraints` | object | Policy flags enforced by validator |
| `thresholds` | object | Quality gates for evaluation metrics |
| `expected` | object | **Hold-out only.** Used for manual eval, not runtime |

### Extra Fields

Any unknown field on flexible models is preserved. Examples the LLM may use:

- `follow_up_days`, `preferred_send_hour`, `scheduled_send_at`
- `amenity_interest`, `city_interest` on profile

---

## Output Record (JSONL line)

Produced by `run.py` or `/api/run`. Schema: `AgentOutput`.

```json
{
  "task_id": "prospect_welcome_day0",
  "should_send": true,
  "next_message": {
    "channel": "sms",
    "send_at": "2025-12-09T09:00:00-06:00",
    "subject": null,
    "body": "Hi Taylor - welcome to Oak Ridge Apartments in Richardson, TX! ...",
    "cta": {
      "type": "schedule_tour",
      "options": ["Thu", "Fri"]
    }
  },
  "next_action": {
    "type": "start_cadence",
    "details": { "name": "prospect_welcome_short_horizon" }
  },
  "reasoning": "SMS is the highest-priority consented channel...",
  "quality": {
    "personalization_score": 0.92,
    "safety_violations": 0,
    "latency_ms": 15
  }
}
```

### Output Rules

| Condition | Rule |
|-----------|------|
| `should_send: false` | `channel` = `none`, `body`/`subject`/`cta` = null |
| `should_send: true`, channel = `email` | `subject` required, non-empty |
| `should_send: true`, channel = `sms`/`push` | `subject` must be null |
| All sends | `body` required when `should_send: true` |
| Opt-out constraint | Body must contain STOP, opt out, or unsubscribe |

### Channel Values

`sms` | `email` | `push` | `none`

### Quality Metrics

| Metric | Range | Description |
|--------|-------|-------------|
| `personalization_score` | 0.0–1.0 | Weighted match against available context |
| `safety_violations` | int ≥ 0 | Blocklist and policy violation count |
| `latency_ms` | int | End-to-end processing time for the record |
