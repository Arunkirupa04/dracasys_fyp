# RRA Methodology

## Protocol version

**rra_v1.0** — frozen in `config/ridge_residual_analysis_config.json`

## Research question

Does Ridge remove all predictable temporal structure from Prophet residuals, or does the remaining Ridge residual still justify a nonlinear learner such as a GRU?

## Data (read-only)

| Item | Value |
|------|-------|
| Source | CSRLE B0-LC frozen parquets |
| Path | `experiments/synthetic_residual_learnability_2026-07-25_164307/data/b0_lc/` |
| Containers | 99 evaluable |
| Hashes | Same as LFHE v1.1 |

## Pipeline

```
CPU (B0-LC synthetic)
    ↓
Prophet (train-only fit, daily seasonality)
    ↓
Prophet residual (validation period)
    ↓
Ridge (96 input → 96 output, α=1.0)
    ↓
Predicted linear residual
    ↓
Remaining Ridge residual = Prophet residual − Ridge prediction
```

## Ridge training (only model trained in this experiment)

| Parameter | Value |
|-----------|-------|
| Estimator | `sklearn.linear_model.Ridge` |
| Alpha | 1.0 (no tuning) |
| Input window | 96 |
| Forecast horizon | 96 |
| Internal split | Chronological 80/20 on train sequences |
| Scaling | Global train residual mean/std |
| Features | `residual_scaled` |
| Target | `residual_scaled` |

Matches CSRLE Stage 2 Ridge baseline protocol (`utils/csrle/stage2_baselines.py`).

## Ridge application at evaluation

Non-overlapping 96-step blocks across the validation period:

1. First block: input = last 96 train Prophet residuals (scaled)
2. Subsequent blocks: input = last 96 actual Prophet residuals (train tail + val prefix)
3. Remaining = actual Prophet residual − Ridge prediction (per step)

## Diagnostics (same family as Residual Pattern Analysis)

| Diagnostic | Scope |
|------------|-------|
| Mean, variance, std, histogram | Validation residuals |
| ACF | Lags 1–50; summary on 1–10 |
| PACF | Yule-Walker; summary on 1–10 |
| Ljung–Box | Lags 10, 20, 30; primary lag 20, α=0.05 |
| FFT / periodogram | Dominant frequency, period, peak power |
| Bootstrap CI | Paired container-level ACF/PACF reduction, seed 12345 |

## Comparison

Prophet validation residual vs Ridge remaining residual — paired per container, cohort aggregates, boxplots, case studies (best/median/worst Ridge Pearson r).

## Exclusions (strict read-only)

- No GRU training
- No Hybrid / Global GRU / peak-aware modification
- No CSRLE / LFHE artifact changes
- No Ridge→GRU architecture implementation
