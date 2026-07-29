# RRAA Methodology

## Scope

Read-only audit of RRA v1.0. No modification of RRA, CSRLE, LFHE, or Hybrid artifacts. New outputs written only under `experiments/ridge_residual_analysis_audit_<timestamp>/`.

## Tasks

| Task | Method |
|------|--------|
| 1 Residual definition | Recompute remaining; verify `remaining = prophet − ridge` sign; detect raw vs z-space mismatch |
| 2 Alignment | Day-1 block: last 96 train z-scores → first 96 val z-scores; compare to sequence builder convention |
| 3 Data splits | Confirm Ridge trained on train only; no train/val timestamp overlap |
| 4 ACF replication | Replicate RRA CSV metrics; recompute on **corrected** z-space remaining |
| 5 Counterintuitive result | Quantify scaling bug + underfit subtraction (R² ≈ 0.06) |
| 6 Linearity | Walk-forward AR(1), AR(2), AR(5) on corrected remaining vs Prophet z-series |
| 7 Decision | Evidence-only answers on Ridge failure mode and Ridge→GRU justification |

## Corrected remaining definition

All operations in **z-scored residual space** (matches Ridge training):

```
prophet_z = (prophet_raw − res_mean) / res_std
ridge_pred_z = Ridge.predict(input_z)
remaining_z = prophet_z − ridge_pred_z
```

RRA v1.0 incorrectly used:

```
remaining_rra = prophet_raw − ridge_pred_z   # BUG
```

## AR evaluation

Per container validation series, walk-forward one-step forecasts from AR(p) with minimum 48-step warm-up. No new forecasting architectures beyond autoregression.

## Decision criteria

| Question | Evidence threshold |
|----------|-------------------|
| Ridge failed as predictor | cohort mean r < 0.20 |
| Ridge removed structure (corrected) | \|ACF\| reduction > 0.01 |
| Remaining predominantly linear | AR(1) forecast r > 0.30 and AR residual Ljung reject < 50% |
| Ridge→GRU justified | NOT if remaining is linear and Ridge barely removes structure |
