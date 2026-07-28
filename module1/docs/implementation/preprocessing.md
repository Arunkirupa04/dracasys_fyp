# Preprocessing

## Responsibilities

1. Validate request fields (length, spacing, NaN, CPU range)
2. Resolve MinMaxScaler (known vs new container)
3. Scale historical CPU to `[0, 1]`

## Input validation (`app/preprocessing/validation.py`)

### Minimum history: 200 steps

Matches production policy for unseen containers. GRU requires 96; Prophet benefits from longer daily context.

### Recommended: 672 steps (7 days)

Improves Prophet daily seasonality estimation.

### Timestamp rules

- ISO-8601 parseable (UTC recommended)
- Strictly increasing
- Uniform **15-minute** spacing (±1 min tolerance)
- No duplicates

### CPU rules

- Finite floats
- Range `[0, 100]` percent
- Standard deviation > 0

## Scaling (`app/preprocessing/scaling.py`)

### Known container

```python
cpu_scaled = scalers[container_id].transform(cpu)
```

Uses frozen scaler from training — **do not refit**.

### New container

```python
scaler = MinMaxScaler().fit(history_cpu)
cpu_scaled = scaler.transform(history_cpu)
```

Fits only on data sent in the request.

## Failure modes

| Error | Cause |
|-------|-------|
| Insufficient history | < 200 points |
| Uneven spacing | Missing/extra samples |
| Degenerate series | Flat CPU line |
| Out of range | CPU < 0 or > 100 |

See [api/schemas.md](../api/schemas.md).
