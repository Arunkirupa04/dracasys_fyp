# Forecast service examples

Sample payloads for the REST API. Full API docs: [docs/api/](../docs/api/README.md).

## Known container (frozen scaler)

| File | Purpose |
|------|---------|
| `sample_request.json` | 200-step history for known container `c_1016` |
| `generate_sample_request.py` | Regenerate synthetic known-container request |

```bash
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d @examples/sample_request.json
```

## Unseen container (validation holdout)

Real Alibaba unseen container **`c_10312`** — history excludes the last 96 steps so you can compare API output to ground truth.

| File | Steps | Purpose |
|------|-------|---------|
| `unseen_c_10312_forecast_request.json` | 672 history | POST body for `/forecast` |
| `unseen_c_10312_validation_ground_truth.json` | 96 actual | Compare predictions vs `actual_cpu` |

```bash
# Regenerate from repo parquet (run from repo root)
python forecast_service/examples/generate_unseen_validation_example.py

# Call API
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d @examples/unseen_c_10312_forecast_request.json

# Compute MAE / RMSE / MAPE (server must be running)
python examples/validate_unseen_forecast.py
```

Reference metrics (pipeline run on 2026-07-27): **MAE 0.35%**, **RMSE 0.47%**, scaler mode `new` (unseen).
