# API Errors

All error responses use this JSON shape:

```json
{
  "status": "error",
  "error_code": "validation_error",
  "message": "Human-readable description",
  "details": {}
}
```

## Status codes

| HTTP | error_code | When |
|------|------------|------|
| 400 | `bad_request` | Unsupported sampling interval, malformed input |
| 422 | `validation_error` | History too short, NaN/Inf, uneven timestamps, zero variance |
| 503 | `service_unavailable` | Artifacts missing or model failed to load at startup |

## Common validation errors (422)

| Message (typical) | Cause | Fix |
|-------------------|-------|-----|
| Insufficient history | Fewer than 200 CPU/timestamp pairs | Send ≥ 200 steps |
| timestamps must be evenly spaced | Gaps or irregular 15-min spacing | Resample to uniform 15-min grid |
| zero variance | Flat/degenerate CPU history | Use a container with varying CPU |
| historical_cpu and timestamps length mismatch | Array length mismatch | Ensure equal-length arrays |
| NaN or infinite values | Invalid floats in `historical_cpu` | Clean data before sending |
| duplicate timestamps | Same timestamp twice | Deduplicate and sort oldest-first |

## Client errors (400)

| Message (typical) | Cause | Fix |
|-------------------|-------|-----|
| Unsupported sampling interval | `sampling_interval_minutes` ≠ 15 | Use `15` |
| Invalid timestamp format | Non-ISO-8601 strings | Use UTC ISO-8601, e.g. `2026-01-01T00:00:00+00:00` |

## Service errors (503)

| Symptom | Cause | Fix |
|---------|-------|-----|
| `model_loaded: false` on `/health` | Artifact path wrong or files corrupt | Check `FORECAST_ARTIFACTS_DIR`, restart service |
| 503 on every forecast | Orchestrator not initialized | Check startup logs, verify TensorFlow + artifacts |

## Batch partial failure

`POST /forecast/batch` returns HTTP **200** even when some items fail. Check each result's `"status"`:

```json
{
  "status": "success",
  "results": [
    { "status": "success", "container_id": "c_1016", "forecast": { ... } },
    { "status": "error", "container_id": "c_bad", "error": "Insufficient history: ..." }
  ],
  "total": 2,
  "succeeded": 1,
  "failed": 1
}
```

## Timeouts

Prophet fits on each request. Allow **≥ 120 seconds** client timeout per forecast. Batch requests scale linearly with container count.

See also [integration.md](integration.md) and [schemas.md](schemas.md).
