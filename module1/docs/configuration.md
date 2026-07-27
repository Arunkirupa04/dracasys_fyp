# Configuration

All settings in `app/config/settings.py`. Override via environment variables with prefix **`FORECAST_`**.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FORECAST_ARTIFACTS_DIR` | `artifacts/hybrid_v1` | Model bundle directory |
| `FORECAST_HOST` | `0.0.0.0` | Server bind address |
| `FORECAST_PORT` | `8000` | Server port |
| `FORECAST_LOG_LEVEL` | `info` | Uvicorn log level |
| `FORECAST_MODEL_VERSION` | `hybrid_v1` | Reported model version |
| `FORECAST_SERVICE_VERSION` | `1.0.0` | API version string |

## Fixed model parameters (not env-configurable without code change)

| Parameter | Value | Source |
|-----------|-------|--------|
| `input_window` | 96 | residual_stats.pkl |
| `default_horizon_steps` | 96 | 24 hours |
| `max_horizon_steps` | 96 | Model output size |
| `sampling_interval_minutes` | 15 | Training data |
| `min_history_steps` | 200 | Production policy |
| `recommended_history_steps` | 672 | 7 days |

## Example `.env`

```env
FORECAST_ARTIFACTS_DIR=/opt/dracasys/artifacts/hybrid_v1
FORECAST_PORT=8080
FORECAST_MODEL_VERSION=hybrid_v1
```

## Docker

Set in `Dockerfile`:

```dockerfile
ENV FORECAST_ARTIFACTS_DIR=/service/artifacts/hybrid_v1
```

Do not mount artifacts read-write.
