# Demo UI API Reference

The FastAPI backend (`ui/app.py`) powers the interactive dashboard. Start it with:

```bash
python start_ui.py
# or
uvicorn ui.app:app --reload --host 0.0.0.0 --port 8080
```

Base URL: `http://localhost:8080`

Interactive OpenAPI docs: `http://localhost:8080/docs`

## Endpoints

### `GET /`

Serves the static dashboard (`ui/static/index.html`).

---

### `GET /api/health`

Health check for load balancers and smoke tests.

**Response**

```json
{ "status": "ok" }
```

---

### `GET /api/sample`

Returns built-in sample records from `data/sample.jsonl`.

**Response**

```json
{
  "filename": "sample.jsonl",
  "record_count": 2,
  "records": [ /* InputRecord objects */ ]
}
```

**Errors:** `500` if sample file cannot be loaded.

---

### `POST /api/upload`

Upload a JSONL file for processing in the UI.

**Request:** `multipart/form-data` with field `file` (must end in `.jsonl`)

**Response**

```json
{
  "filename": "my_data.jsonl",
  "record_count": 10,
  "records": [ /* parsed InputRecord dicts */ ]
}
```

**Errors:**

| Status | Cause |
|--------|-------|
| `400` | Not a `.jsonl` file, invalid JSON, schema error, or empty file |

---

### `POST /api/run`

Run the agent on a list of input records.

**Request body**

```json
{
  "records": [ /* array of InputRecord-compatible objects */ ],
  "mock": true
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `records` | `array` | required | Input records to process |
| `mock` | `boolean` | `true` | Use offline mock LLM (no API key needed) |

**Response**

```json
{
  "outputs": [ /* AgentOutput objects */ ],
  "trace": {
    "run_id": "uuid",
    "mock": true,
    "provider": "openai",
    "model": "gpt-4o-mini",
    "started_at": "2025-12-09T12:00:00+00:00",
    "total_latency_ms": 45,
    "summary": {
      "total_records": 2,
      "sent": 2,
      "suppressed": 0,
      "channel_distribution": { "sms": 1, "email": 1 },
      "average_personalization_score": 0.92,
      "max_safety_violations": 0,
      "average_latency_ms": 22.5,
      "threshold_pass_rate": 1.0
    },
    "records": [ /* RecordTrace with pipeline steps */ ]
  }
}
```

**Trace step phases:** `ingest`, `prompt`, `llm`, `parse`, `validate`, `retry`, `evaluate`, `threshold`, `complete`, `error`

**Errors:**

| Status | Cause |
|--------|-------|
| `400` | Empty records or invalid schema |
| `500` | LLM invocation failure |

---

### `GET /static/*`

Static assets (CSS, JS, sample-data.json).

## Example: Run Sample via curl

```bash
# Fetch sample records
curl -s http://localhost:8080/api/sample | jq '.records' > records.json

# Run agent in mock mode
curl -s -X POST http://localhost:8080/api/run \
  -H "Content-Type: application/json" \
  -d "{\"records\": $(cat records.json), \"mock\": true}" | jq '.summary'
```

## CORS

The API allows all origins (`allow_origins=["*"]`) for local demo use. Restrict this in production deployments.
