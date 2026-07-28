# API Examples

Sample payloads and client code for calling the forecast API.

## Sample request files

| File | Container | History steps | Purpose |
|------|-----------|---------------|---------|
| [sample_request.json](../../examples/sample_request.json) | `c_1016` (known) | 200 | Minimal valid request |
| [unseen_c_10312_forecast_request.json](../../examples/unseen_c_10312_forecast_request.json) | `c_10312` (unseen) | 672 | Real holdout — last 96 steps withheld |
| [unseen_c_10312_validation_ground_truth.json](../../examples/unseen_c_10312_validation_ground_truth.json) | `c_10312` | 96 actual | Compare predictions for MAE/RMSE |

Generate synthetic known-container data:

```bash
python examples/generate_sample_request.py
```

Regenerate unseen holdout examples from repo parquet:

```bash
python examples/generate_unseen_validation_example.py
python examples/generate_unseen_validation_example.py --container-id c_11013
```

## curl

```bash
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d @examples/sample_request.json
```

Full script: [examples/client_curl.sh](../../examples/client_curl.sh)

## Python

```python
import httpx

payload = {
    "container_id": "c_1016",
    "historical_cpu": [...],  # >= 200 floats, oldest first
    "timestamps": [...],      # ISO-8601 UTC, same length
    "sampling_interval_minutes": 15,
    "prediction_horizon_steps": 96,
}

response = httpx.post("http://localhost:8000/forecast", json=payload, timeout=120.0)
response.raise_for_status()
data = response.json()
print(data["forecast"]["predicted_cpu_percent"][:3])
```

Full script: [examples/client_python.py](../../examples/client_python.py)

## Java / JavaScript

| Language | File |
|----------|------|
| Java | [examples/client_java.java](../../examples/client_java.java) |
| JavaScript | [examples/client_javascript.js](../../examples/client_javascript.js) |

## Validate against ground truth

For unseen container `c_10312`, compare API output to held-out actuals:

```bash
python examples/validate_unseen_forecast.py
```

Computes MAE, RMSE, and MAPE against `unseen_c_10312_validation_ground_truth.json`.

## Minimal request shape

```json
{
  "container_id": "c_1016",
  "historical_cpu": [12.5, 13.1],
  "timestamps": ["2026-01-01T00:00:00+00:00", "2026-01-01T00:15:00+00:00"],
  "sampling_interval_minutes": 15,
  "prediction_horizon_steps": 96
}
```

Field definitions: [schemas.md](schemas.md). Endpoint details: [reference.md](reference.md).
