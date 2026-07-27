# System Design

## Design goals

1. **Inference only** — artifacts loaded at startup, never modified
2. **Integration-friendly** — JSON in/out, OpenAPI/Swagger docs
3. **Fail clearly** — validation errors before expensive Prophet/GRU work where possible
4. **Research isolation** — zero imports from thesis codebase

## Request lifecycle

```
POST /forecast
    │
    ├─► Pydantic schema validation (types, ranges)
    │
    ├─► Business validation (history length, spacing, NaN)
    │
    ├─► Scaler resolution
    │       known container → frozen scalers.pkl
    │       new container   → fit MinMax on provided history
    │
    ├─► Prophet fit (history) + forecast (future timestamps)
    │
    ├─► Residual window (last 96) → GRU → inverse z-score
    │
    ├─► Combine Prophet + GRU → inverse MinMax → CPU %
    │
    └─► JSON response + metadata
```

## Startup lifecycle

```
Application start
    │
    ├─► Load settings (env vars)
    ├─► Verify artifact files exist
    ├─► Load model.keras (TensorFlow)
    ├─► Load scalers.pkl, residual_stats.pkl
    └─► Create ForecastOrchestrator → app.state
```

If artifact loading fails, `/health` returns `unhealthy`.

## Scaler modes

| Mode | When | Scaler source |
|------|------|---------------|
| `known` | `container_id` in training cohort | `scalers.pkl` (frozen) |
| `new` | Unknown container ID | MinMax fit on request history |

Global residual z-score (`res_mean`, `res_std`) is **always** from `residual_stats.pkl` — never refit.

## Error strategy

| Condition | HTTP | Code |
|-----------|------|------|
| Malformed JSON / schema | 422 | validation_error |
| Wrong interval, bad timestamps | 400 / 422 | bad_request / validation_error |
| History too short | 422 | validation_error |
| Missing artifacts at startup | 503 | service_unavailable |

## Extension points

- Authentication middleware (FastAPI dependencies)
- Async batch queue for large batches
- Caching Prophet models per container (not implemented — stateless by design)
- Metrics endpoint (Prometheus) — see [future_extensions.md](future_extensions.md)
