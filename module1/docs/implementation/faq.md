# Implementation FAQ

Questions about deployment, artifacts, and the inference pipeline. For API usage, see [api/faq.md](../api/faq.md).

## Service

**Q: Does this service train models?**  
A: No. Inference only. Artifacts are frozen.

**Q: Where did the model come from?**  
A: Production Hybrid training on 435 Alibaba trace containers. Artifacts copied to `artifacts/hybrid_v1/`.

**Q: Is this the same as the research Hybrid?**  
A: Same frozen methodology and weights as production. Research experiments are separate and not used at runtime.

## Artifacts

**Q: Can I update the model without redeploying code?**  
A: Yes. Place a new bundle under `artifacts/`, set `FORECAST_ARTIFACTS_DIR`, and restart. See [deployment.md](deployment.md).

**Q: What files must be in the artifact bundle?**  
A: See [artifact_reference.md](artifact_reference.md).

## Data constraints

**Q: Can I use 5-minute data?**  
A: No. Model trained on 15-minute intervals. Resample client-side before calling the API.

**Q: Why 200 minimum history steps?**  
A: Matches production unseen-container policy and ensures Prophet + GRU have sufficient context.

## Operations

**Q: 503 service unavailable?**  
A: Artifacts missing or model failed to load. Check `FORECAST_ARTIFACTS_DIR` and logs.

**Q: How long does startup take?**  
A: Allow 30–60s for TensorFlow model load. See [deployment.md](deployment.md).

**Q: How do other modules call this?**  
A: HTTP POST to `/forecast`. See [api/integration.md](../api/integration.md).
