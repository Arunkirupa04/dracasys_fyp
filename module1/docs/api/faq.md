# API FAQ

## General

**Q: Does this API train models?**  
A: No. It serves a frozen Hybrid Prophet+GRU model for inference only.

**Q: What data format does the API expect?**  
A: JSON with aligned `historical_cpu` (percent) and `timestamps` (ISO-8601 UTC) at 15-minute intervals.

## Requests

**Q: Why 200 minimum history steps?**  
A: Production policy for unseen containers; ensures Prophet and GRU have sufficient context.

**Q: Should I send 672 steps?**  
A: Recommended (7 days). Minimum is 200; more history generally improves Prophet fit.

**Q: What if my container ID is new?**  
A: The service fits a new MinMaxScaler on your history (`scaler_mode: "new"`). GRU residual stats remain global.

**Q: What if my container ID is in the training set?**  
A: Known IDs (435 containers) use the frozen scaler from training (`scaler_mode: "known"`).

**Q: Can I forecast 48 hours?**  
A: No. Maximum horizon is 96 steps (24 hours at 15-min sampling).

**Q: Can I use 5-minute data?**  
A: No. Resample to 15-minute intervals before calling the API.

**Q: Do timestamps need timezone info?**  
A: Use ISO-8601 with timezone (UTC recommended). Naive UTC strings are accepted.

## Responses

**Q: What is `prophet_component_percent`?**  
A: The seasonal baseline from Prophet — a decomposition output, not a separate API.

**Q: Are prediction intervals / confidence bands provided?**  
A: No. The model returns point forecasts only.

## Integration

**Q: How do other DracaSys modules call this?**  
A: HTTP POST to `/forecast`. See [integration.md](integration.md).

**Q: Is there a Python SDK?**  
A: No official SDK. Use `httpx` or `requests` — see [examples.md](examples.md).

**Q: Is authentication built in?**  
A: No. Place the service behind an API gateway or reverse proxy for production.

**Q: Is there rate limiting?**  
A: Not built-in. Add at the gateway if needed.

## Errors

**Q: 503 service unavailable?**  
A: Artifacts missing or model failed to load. Check `/health` and `FORECAST_ARTIFACTS_DIR`.

**Q: 422 insufficient history?**  
A: Send at least 200 aligned CPU/timestamp pairs.

See [errors.md](errors.md) for the full troubleshooting table.

## Model background

**Q: Is this the same as the research Hybrid?**  
A: Same frozen methodology and weights as production. Research experiments are separate and not used at runtime.

For deployment and artifact details, see [implementation/faq.md](../implementation/faq.md).
