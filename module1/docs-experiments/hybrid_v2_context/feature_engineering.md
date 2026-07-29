# Feature Engineering

All features computed **causally** from train-period information at inference.

## Channel definitions

| Feature | Formula | Normalization |
|---------|---------|---------------|
| `residual_scaled` | `(residual - μ_res) / σ_res` | Global train residual stats |
| `prophet_yhat_scaled` | `(prophet_pred - μ_yhat) / σ_yhat` | Global train Prophet level stats |
| `res_roll_std_12` | Rolling std of `residual_scaled`, w=12, `min_periods=1` | None (already scaled space) |
| `cpu_std` | Container train std of scaled CPU | Broadcast constant per container |
| `is_peak_p90` | `1 if cpu_scaled >= P90_train(container)` | Binary |

## Leakage controls

- Prophet fit on **train only**; validation Prophet forecast uses known future timestamps only for `yhat`, not for GRU input window
- GRU input window = **last 96 train steps** before Day-1 forecast
- P90 thresholds from **train cpu_scaled** per container
- No validation CPU in input window features
- Rolling std uses **past-only** windows (no centering)

## Validation checks

Automated per variant in `verification/feature_leakage_checks.json`:

- All features present
- No NaN in feature matrix
- Volatility non-negative
- Peak flag binary

## Implementation

`utils/hcerl/features.py` — `prepare_training_frame()`, `enrich_inference_train_container()`
