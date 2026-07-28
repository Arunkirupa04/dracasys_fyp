# Prophet Pipeline

## Role

Prophet models the **seasonal baseline** of CPU utilisation. The GRU then forecasts the **residual** Prophet cannot explain.

## Per-request behaviour

Prophet is **fit fresh on each request** using the client's historical window:

```python
Prophet(daily_seasonality=True, weekly_seasonality=False)
model.fit(history_timestamps, cpu_scaled)
forecast = model.predict(future_timestamps)
```

This matches the frozen production inference protocol.

## Why refit every request

- Service is stateless — no stored Prophet models
- New containers must work without pre-trained Prophet weights
- Known containers also refit on latest history for deployment freshness

## Output

- `prophet_component_percent` — inverse-transformed seasonal forecast for each future step
- Used as additive component: `final = prophet + gru_residual`

## Configuration

From `app/config/settings.py`:

| Setting | Default |
|---------|---------|
| `prophet_daily_seasonality` | true |
| `prophet_weekly_seasonality` | false |

## Performance

Prophet Stan sampling is the main latency driver (typically seconds per request).

## Limitations

- Requires sufficient history for daily pattern (~2+ days minimum; 7 days recommended)
- Assumes 15-minute sampling aligned with training data

See [forecast_pipeline.md](forecast_pipeline.md).
