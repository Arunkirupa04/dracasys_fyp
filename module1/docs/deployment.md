# Deployment

## Local

```bash
cd forecast_service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

Service listens on `http://0.0.0.0:8000`.

## Docker

```bash
docker build -t dracasys-forecast:1.0.0 .
docker run --rm -p 8000:8000 dracasys-forecast:1.0.0
```

## Production recommendations

| Concern | Recommendation |
|---------|----------------|
| HTTPS | Terminate TLS at nginx/traefik |
| Resources | ≥ 2 GB RAM; Prophet is memory-intensive |
| Startup | Allow 30–60s for TensorFlow model load |
| Health checks | Poll `GET /health` |
| Artifacts | Bake into image or mount read-only volume |
| Scaling | Horizontal replicas (stateless) |
| Timeouts | Client timeout ≥ 120s per forecast |

## Verify deployment

```bash
curl -f http://localhost:8000/health
curl -f http://localhost:8000/model/info
python examples/generate_sample_request.py
curl -f -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d @examples/sample_request.json
```

## Updating the model

1. Obtain new frozen artifact bundle (`hybrid_v2`)
2. Place in `artifacts/hybrid_v2/`
3. Update `FORECAST_ARTIFACTS_DIR` and `FORECAST_MODEL_VERSION`
4. Restart service
5. Re-run verification

Never retrain inside the running container.

## Separation from research

This deployment does **not** use:

- `experiments/`
- Research notebooks
- `production/scripts/train_production_hybrid.py` at runtime

Training remains a separate offline process.
