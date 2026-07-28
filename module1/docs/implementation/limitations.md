# Limitations

## Model

- **Horizon:** Maximum 96 steps (24 hours at 15-min sampling)
- **Sampling:** Fixed 15-minute intervals only
- **Uncertainty:** No confidence intervals provided
- **Spikes:** MSE-trained model may under-forecast CPU bursts
- **Prophet ceiling:** ~78% variance explained; GRU adds incremental value

## Service

- **Latency:** 5–30+ seconds per request (Prophet fit)
- **Stateless:** Client must send full history each call
- **Batch:** Sequential processing (not parallel)
- **Auth:** Not included — add at gateway
- **Rate limiting:** Not included

## Data

- **Minimum history:** 200 steps (~50 hours)
- **CPU range:** 0–100 percent assumed
- **Missing data:** Not supported — gaps must be pre-filled by client
- **Multi-metric:** CPU only (no memory, network, etc.)

## Containers

- **435 known IDs** have frozen scalers
- **New containers** fit scaler on provided history — quality depends on history length
- Container IDs not in training still work but may differ from research eval protocol

## Operational

- TensorFlow + Prophet increase image size (~2 GB+)
- First request after cold start may be slower
- Artifact corruption → 503 until restart with valid bundle

See [future_extensions.md](future_extensions.md) for planned improvements.
