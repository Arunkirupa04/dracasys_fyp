# DracaSys Hybrid CPU Forecast Service

Production-ready REST API exposing the **frozen Hybrid Prophet+GRU model** for 24-hour CPU utilisation forecasting.

This is a **deployment project** — not research, not training, **inference only**.

## Quick start

```bash
cd forecast_service
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Generate a sample request (>= 200 history steps)
python examples/generate_sample_request.py

# Start API (loads artifacts at startup — ~30s first time)
python run.py
```

Open **http://localhost:8000/docs** for interactive Swagger UI.

```bash
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d @examples/sample_request.json
```

## Project layout

```
forecast_service/
├── app/                 # Application code (FastAPI)
├── artifacts/hybrid_v1/ # Frozen model bundle (DO NOT retrain)
├── docs/                # Structured documentation
├── examples/            # Client examples (curl, Python, Java, JS)
├── tests/               # Unit and integration tests
├── requirements.txt
├── Dockerfile
└── run.py
```

## Configuration

Environment variables (prefix `FORECAST_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `FORECAST_ARTIFACTS_DIR` | `artifacts/hybrid_v1` | Model bundle path |
| `FORECAST_HOST` | `0.0.0.0` | Bind address |
| `FORECAST_PORT` | `8000` | Port |
| `FORECAST_MODEL_VERSION` | `hybrid_v1` | Reported model version |

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| GET | `/version` | Service and model version |
| GET | `/model/info` | Model metadata |
| POST | `/forecast` | Single-container forecast |
| POST | `/forecast/batch` | Batch forecast (max 50) |

## Documentation

| Section | Description |
|---------|-------------|
| **[docs/api/](docs/api/README.md)** | **API documentation** — endpoints, schemas, integration, errors, examples |
| **[docs/implementation/](docs/implementation/README.md)** | Deployment, architecture, pipeline, artifacts, development |
| **[docs/README.md](docs/README.md)** | Full documentation index |

## Isolation

- Does **not** import from `experiments/`, `production/`, research `utils/`, or notebooks
- Consumes only copied artifacts under `artifacts/hybrid_v1/`
- Never retrains Prophet or GRU

## Docker

```bash
docker build -t dracasys-forecast .
docker run -p 8000:8000 dracasys-forecast
```
