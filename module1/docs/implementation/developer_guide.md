# Developer Guide

## Setup

```bash
cd forecast_service
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run with auto-reload (development)

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Working directory must be `forecast_service/` (so `app` package resolves).

## Project conventions

| Rule | Rationale |
|------|-----------|
| No imports from repo root `utils/`, `production/`, `experiments/` | Research isolation |
| Artifacts read-only | Inference-only service |
| Validation before Prophet | Fail fast |
| Pydantic schemas for API | OpenAPI + type safety |

## Adding an endpoint

1. Define schema in `app/api/schemas.py`
2. Add route in `app/api/routes.py`
3. Implement logic in `app/services/`
4. Document in `docs/api/reference.md`
5. Add tests in `tests/`

## Running tests

```bash
cd forecast_service
pytest tests/ -v
```

Validation tests run without TensorFlow. Full inference tests require artifacts + TF.

## Code layout

```
app/
  main.py           # FastAPI app factory
  api/              # HTTP layer
  services/         # Business logic
  inference/        # Hybrid pipeline
  preprocessing/    # Validation + scaling
  prophet/          # Prophet wrapper
  config/           # Settings
```

## Debugging

- Swagger UI: `/docs`
- Check artifact paths: `GET /health`
- Prophet verbose: set Stan log level in prophet forecaster (dev only)

See [testing.md](testing.md).
