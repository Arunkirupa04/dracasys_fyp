# Future Extensions

Planned improvements (not yet implemented):

## API

- [ ] Async batch queue with job ID polling
- [ ] WebSocket streaming for long batches
- [ ] Authentication (API keys / JWT via FastAPI dependencies)
- [ ] Prometheus `/metrics` endpoint
- [ ] OpenAPI webhook callbacks

## Model

- [ ] `hybrid_v2` artifact release pipeline
- [ ] Optional prediction intervals (if retrained with uncertainty)
- [ ] Configurable horizon if model retrained

## Performance

- [ ] Prophet model caching per container ID
- [ ] ONNX/TF Serving for GRU only
- [ ] Request coalescing for batch workloads

## Integration

- [ ] Official Python client package on PyPI
- [ ] Kubernetes Helm chart
- [ ] Health check sub-endpoint for Prophet-only smoke test

## Out of scope

- Online learning / retraining in API
- Importing research experiment code
- Multi-variate forecasting (memory, disk, network)

Contributions should maintain **research isolation** and **inference-only** policy.
