# LFHE Training

## Frozen configuration

See `experiments/loss_function_hypothesis_2026-07-26_130252/config/lfhe_config_frozen.json`.

| Parameter | Value |
|-----------|-------|
| Loss (control) | MSE |
| Loss (treatment) | DA-MSE, λ=1.0, ε=1e−6 |
| Optimizer | Adam |
| Batch size | 64 |
| Max epochs | 100 |
| Early stopping | val_loss, patience=10 |
| GRU training seed | None (CSRLE Stage 2 parity) |
| Bootstrap seed | 12345 |

## MSE control arm (`b0_lc_mse`)

| Field | Value |
|-------|-------|
| Sequences | 41,650 (33,320 / 8,330) |
| Epochs run | See `training_metadata.json` |
| res_mean | ~2.1e−5 |
| res_std | ~0.110 |

## DA-MSE treatment arm (`b0_lc_da_mse`)

| Field | Value |
|-------|-------|
| Same data / architecture as MSE | Yes |
| Only change | DA-MSE loss |

Training histories: `*/training/training_history.csv`  
Loss curves: `plots/training_loss_comparison.{png,pdf}`

## DA-MSE implementation

```text
L = MSE + λ · max(0, log(σ_y) − log(σ_ŷ))²
```

Per-sequence σ over 96-step horizon, ddof=0, one-sided under-dispersion penalty only.

Unit tests verified in `tests/test_lfhe_loss.py` (inline run; no pytest dependency).
