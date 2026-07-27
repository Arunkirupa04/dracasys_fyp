# FAQ

## General

**Q: Does this service train models?**  
A: No. Inference only. Artifacts are frozen.

**Q: Where did the model come from?**  
A: Production Hybrid training on 435 Alibaba trace containers. Artifacts copied to `artifacts/hybrid_v1/`.

**Q: Can I use 5-minute data?**  
A: No. Model trained on 15-minute intervals. Resample client-side.

## Requests

**Q: Why 200 minimum history steps?**  
A: Matches production unseen-container policy and ensures Prophet + GRU have sufficient context.

**Q: What if my container ID is new?**  
A: Service fits a new MinMaxScaler on your history. GRU residual stats remain global.

**Q: Can I forecast 48 hours?**  
A: No. Maximum horizon is 96 steps (24 hours).

## Errors

**Q: 503 service unavailable?**  
A: Artifacts missing or model failed to load. Check `FORECAST_ARTIFACTS_DIR` and logs.

**Q: 422 insufficient history?**  
A: Send at least 200 aligned CPU/timestamp pairs.

## Integration

**Q: How do other modules call this?**  
A: HTTP POST to `/forecast`. See [integration_guide.md](integration_guide.md).

**Q: Is there a Python SDK?**  
A: Use `httpx` or `requests` — see `examples/client_python.py`.

## Research

**Q: Is this the same as the research Hybrid?**  
A: Same frozen methodology and weights as production. Research experiments are separate and not used at runtime.
