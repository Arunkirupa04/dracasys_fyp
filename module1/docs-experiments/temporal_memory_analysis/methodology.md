# Methodology

## Protocol version

**tma_v1.0** — frozen in `config/temporal_memory_analysis_config.json`

## Analytical principle: three distinct signals

TMA analyzes **three separate time series per container**. Each series corresponds to a different stage of the forecasting pipeline and must be interpreted independently.

```
Signal 1: Original CPU          →  Global GRU input (raw scaled CPU)
Signal 2: Prophet-equivalent residual  →  Hybrid GRU input (after Prophet)
Signal 3: Hybrid post-forecast residual  →  Hybrid forecast error (actual − forecast)
```

**Rule:** Window-sufficiency and long-lag memory conclusions from Signal 1 (CPU) apply to **Global GRU**. They do **not** transfer to **Hybrid GRU**, which operates on Signal 2. Signal 3 assesses forecast-error structure, not input-window adequacy.

Mixing these signals — for example, citing CPU long-memory as evidence that Hybrid needs a longer window — is **methodologically incorrect**, because the Hybrid GRU never receives original CPU.

## Cohort and temporal span

| Item | Value |
|------|-------|
| Containers | 99 evaluable (same as CSRLE/LFHE/RRA) |
| Cohort definition | Validation-evaluable containers from `selected_containers.npy` |
| CPU / Prophet-equivalent span | Full chronological train + validation (~768 steps) |
| Hybrid post-forecast span | Validation day-1 only (96 steps, frozen cache) |
| Maximum lag | 672 (7 days × 96 steps) |

Validation-only series (~154 steps) cannot support 672-lag ACF; full train+val timeline is used for long-memory analysis on CPU and Prophet-equivalent residuals while keeping the validation cohort.

## Series analyzed

### Signal 1: Original CPU

Real CPU utilization (%) from frozen `train_df.parquet` and `val_df.parquet`, inverse-transformed with frozen `scalers.pkl`. No model fitting.

**Interpretation scope:** Global GRU — models that forecast directly from scaled CPU sequences.

### Signal 2: Prophet-equivalent residual (read-only)

Because `Prophet.fit()` is forbidden in this experiment, residuals are reconstructed via:

- Additive daily Fourier seasonality (10 harmonics, period = 96 steps)
- Linear trend
- OLS fit on **train CPU only**, applied to train and validation segments

This mirrors Hybrid Prophet configuration (`daily_seasonality=True`, `weekly_seasonality=False`) without invoking Prophet.

**Interpretation scope:** Hybrid GRU — the signal the GRU receives during training and inference after Prophet decomposition.

### Signal 3: Hybrid post-forecast residual (read-only)

From frozen baseline inference cache:

```
experiments/peak_aware_2026-07-14_164518/evaluation/baseline_inference_cache.pkl
```

Definition: `actual_day1_real - day1_final_real` (96 validation steps per container).

**Interpretation scope:** Hybrid forecast quality — whether structured errors remain after the full Prophet + GRU pipeline on the day-1 evaluation horizon.

## Diagnostics

| Analysis | Details |
|----------|---------|
| Long-lag ACF/PACF | Up to 672 lags; cohort mean, median, 95% CI |
| Daily periodicity | Lags 96, 192, 288, 384, 480, 576, 672; bootstrap CI; threshold fractions (0.1, 0.2, 0.3) |
| FFT / PSD | Periodogram per container; cohort-mean Welch PSD |
| Memory length | First insignificant lag, half-life, decorrelation (\|ACF\|<0.1), integrated autocorrelation time |
| Window sufficiency | Cumulative squared-ACF capture at 96, 192, 288, 672 lags |
| Hypothetical windows | Mean \|ACF\| within 96, 192, 288, 384 (information proxy only) |
| Paired tests | Mean \|ACF\| lags 1–96 vs beyond 96; Wilcoxon signed-rank; Cohen's d_z; bootstrap CI |

## Bootstrap

| Parameter | Value |
|-----------|-------|
| Seed | 12345 |
| Resamples | 2000 |
| CI level | 95% |

## Interpretation boundaries

TMA quantifies **available temporal autocorrelation** at different window lengths. It does **not**:

- Retrain or evaluate forecasting models
- Compare MAE/RMSE under alternative windows
- Prove that longer windows improve accuracy
- Recommend longer windows for Hybrid (residual signal has short memory)

See [discussion.md](discussion.md) for architecture-specific interpretation (Global vs Hybrid).

## Exclusions (strict read-only)

- No Hybrid / Global GRU / Peak-Aware / CSRLE / LFHE / RRA / Audit modification
- No Prophet.fit(), GRU fitting, Ridge fitting
- No new datasets
