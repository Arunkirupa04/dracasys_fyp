# Request & Response Schemas

## POST /forecast — Request

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `container_id` | string | yes | — | Container identifier |
| `historical_cpu` | float[] | yes | — | CPU % values, oldest first |
| `timestamps` | string[] | yes | — | ISO-8601 UTC, same length as CPU |
| `sampling_interval_minutes` | int | no | 15 | Must be **15** |
| `prediction_horizon_steps` | int | no | 96 | Forecast length (1–96) |

### Validation rules

| Rule | Value |
|------|-------|
| Minimum history | **200** steps |
| Recommended history | **672** steps (7 days) |
| CPU range | 0–100 (percent) |
| Timestamp format | ISO-8601 with timezone (UTC recommended) |
| Spacing | Uniform 15 minutes ± 1 minute tolerance |
| Uniqueness | No duplicate timestamps |
| NaN / Inf | Rejected |

### Container assumptions

- **Known ID** (435 training containers): uses frozen `scalers.pkl` entry
- **Unknown ID**: fits new MinMaxScaler on provided history only
- Global GRU residual stats always from `residual_stats.pkl`

## POST /forecast — Response (200)

```json
{
  "status": "success",
  "container_id": "string",
  "metadata": {
    "model_version": "hybrid_v1",
    "processing_time_ms": 0.0,
    "scaler_mode": "known | new",
    "history_steps_used": 0,
    "horizon_steps": 96,
    "sampling_interval_minutes": 15
  },
  "forecast": {
    "timestamps": ["ISO-8601..."],
    "predicted_cpu_percent": [0.0],
    "prophet_component_percent": [0.0],
    "gru_residual_component_percent": [0.0]
  }
}
```

| Output field | Unit | Description |
|--------------|------|-------------|
| `predicted_cpu_percent` | % | Final Hybrid forecast |
| `prophet_component_percent` | % | Seasonal baseline component |
| `gru_residual_component_percent` | % | Learned residual adjustment |
| `timestamps` | ISO-8601 | Future times, first step = last history + 15 min |

Arrays are **ordered chronologically** with length = `horizon_steps`.

## Error response (4xx / 5xx)

```json
{
  "status": "error",
  "error_code": "validation_error",
  "message": "Human-readable description",
  "details": {}
}
```

| HTTP | error_code | Typical cause |
|------|------------|---------------|
| 400 | bad_request | Invalid interval, malformed input |
| 422 | validation_error | Short history, NaN, spacing |
| 503 | service_unavailable | Missing/corrupt artifacts |

## POST /forecast/batch

Request: `{ "requests": [ ForecastRequest, ... ] }` (max 50)

Response:

```json
{
  "status": "success",
  "results": [
    { "status": "success", "container_id": "...", "forecast": {...} },
    { "status": "error", "container_id": "...", "error": "..." }
  ],
  "total": 2,
  "succeeded": 1,
  "failed": 1
}
```

## Confidence intervals

The production model does **not** provide prediction intervals. `prophet_component_percent` and `gru_residual_component_percent` are decomposition outputs, not uncertainty bounds.
