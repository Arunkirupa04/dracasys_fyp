# Versioning

## Service version

`FORECAST_SERVICE_VERSION` — API package semver (currently `1.0.0`).

Reported by `GET /version`.

## Model version

`FORECAST_MODEL_VERSION` — artifact bundle ID (currently `hybrid_v1`).

Increment when deploying a new trained model:

| Version | Description |
|---------|-------------|
| `hybrid_v1` | Initial frozen production model (435 containers) |
| `hybrid_v2` | Future retrain (not yet released) |

## API version

URL path versioning not used. OpenAPI `api_version: v1` in `/version` response.

Breaking API changes should bump service major version and be documented.

## Artifact compatibility

Each model version requires matching:

- `model.keras`
- `scalers.pkl`
- `residual_stats.pkl`
- `production_config.json`

Do not mix files across versions.

## Changelog policy

Document in deployment release notes:

- Model version
- Mean Day-1 MAE on eval cohort (from offline eval, not API)
- Any API schema changes
