# API Reference

Base URL: `http://<host>:<port>` (default `http://localhost:8000`)

All forecast endpoints accept and return `application/json`.

OpenAPI (auto-generated): `/docs` · `/openapi.json`

---

## GET /

Service discovery.

**Response 200**

```json
{
  "service": "dracasys-hybrid-forecast",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health"
}
```

---

## GET /health

Liveness check and model load status. Use for orchestration probes.

**Response 200**

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `healthy` if model loaded, else `unhealthy` |
| `model_loaded` | bool | Artifacts loaded successfully |
| `model_version` | string \| null | e.g. `hybrid_v1` |
| `artifacts_dir` | string | Resolved artifact bundle path |

**Example**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "hybrid_v1",
  "artifacts_dir": "artifacts/hybrid_v1"
}
```

---

## GET /version

Service and model version strings.

**Response 200**

| Field | Type | Description |
|-------|------|-------------|
| `service_version` | string | API package version |
| `model_version` | string | Frozen model bundle ID |
| `api_version` | string | API contract version (`v1`) |

**Example**

```json
{
  "service_version": "1.0.0",
  "model_version": "hybrid_v1",
  "api_version": "v1"
}
```

---

## GET /model/info

Model capabilities and constraints. Call before integration to read history requirements.

**Response 200**

| Field | Type | Description |
|-------|------|-------------|
| `model_version` | string | Frozen bundle ID |
| `model_type` | string | e.g. `hybrid_prophet_gru` |
| `input_window` | int | GRU input window (96) |
| `default_horizon_steps` | int | Default forecast length (96) |
| `max_horizon_steps` | int | Maximum forecast length (96) |
| `minimum_history_steps` | int | Minimum required history (200) |
| `recommended_history_steps` | int | Recommended history (672) |
| `sampling_interval_minutes` | int | Required interval (15) |
| `known_containers_count` | int | Containers with frozen scalers (435) |
| `residual_normalization` | string | e.g. `global` |
| `prophet` | object | Prophet configuration summary |
| `architecture_summary` | string | GRU layer summary |

---

## POST /forecast

Single-container 24-hour CPU forecast.

**Request body:** [schemas.md](schemas.md)

**Response 200:** [schemas.md](schemas.md)

**Errors:** 400, 422, 503 — see [errors.md](errors.md)

**Example request**

```bash
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d @examples/sample_request.json
```

**Example response (truncated)**

```json
{
  "status": "success",
  "container_id": "c_1016",
  "metadata": {
    "model_version": "hybrid_v1",
    "processing_time_ms": 8421.5,
    "scaler_mode": "known",
    "history_steps_used": 200,
    "horizon_steps": 96,
    "sampling_interval_minutes": 15
  },
  "forecast": {
    "timestamps": ["2026-01-03T02:00:00+00:00"],
    "predicted_cpu_percent": [14.2],
    "prophet_component_percent": [13.8],
    "gru_residual_component_percent": [0.4]
  }
}
```

---

## POST /forecast/batch

Up to **50** single-container forecasts in one request. Partial success allowed.

**Request body**

```json
{
  "requests": [
    {
      "container_id": "c_1016",
      "historical_cpu": [12.5, 13.1],
      "timestamps": ["2026-01-01T00:00:00+00:00", "2026-01-01T00:15:00+00:00"],
      "sampling_interval_minutes": 15,
      "prediction_horizon_steps": 96
    }
  ]
}
```

**Response 200**

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Always `success` at batch level |
| `results` | array | Per-container result or error |
| `total` | int | Number of requests processed |
| `succeeded` | int | Successful forecasts |
| `failed` | int | Failed forecasts |

Each successful result item includes `forecast` (same shape as `/forecast`). Failed items include `"status": "error"` and an `error` message string.

See [schemas.md](schemas.md#post-forecastbatch) for full schema.

---

## Related

- [schemas.md](schemas.md) — field definitions and validation rules
- [integration.md](integration.md) — integration workflow
- [examples.md](examples.md) — sample payloads and client code
- [errors.md](errors.md) — error codes and troubleshooting
