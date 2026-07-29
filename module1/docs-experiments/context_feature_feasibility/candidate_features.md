# Task 2 — Candidate Contextual Features

Complete catalogue of features that could augment the Hybrid GRU input beyond `residual_scaled`. Grouped by type. See [feature_analysis.md](feature_analysis.md) for the six evaluation criteria per feature.

**Legend:** ✅ recommended for consideration · ⚠️ marginal · ❌ not recommended

---

## A. Temporal / calendar features

| Feature | Definition | Online? |
|---------|------------|---------|
| `hour` | Hour of day (0–23) | ✅ |
| `hour_sin`, `hour_cos` | Cyclical encoding of hour | ✅ |
| `minute_of_day` | `hour*60 + minute` | ✅ |
| `minute_sin`, `minute_cos` | Cyclical 96-step intraday phase | ✅ |
| `day_index` | Floor(step / 96) since series start | ✅ |
| `step_index` | Absolute timestep index | ✅ |
| `time_since_start` | Elapsed time from first observation | ✅ |
| `weekday` | 0=Mon … 6=Sun | ✅ |
| `is_weekend` | Binary weekend flag | ✅ |
| `day_of_month` | Calendar day | ✅ (weak for 6-day train) |

---

## B. Residual history transforms (deterministic functions of GRU input)

| Feature | Definition | Online? |
|---------|------------|---------|
| `res_lag_k` | Residual at t−k | ✅ |
| `res_diff1` | r(t) − r(t−1) | ✅ |
| `res_diff2` | Second difference | ✅ |
| `res_roll_mean_w` | Rolling mean of residuals, window w | ✅ |
| `res_roll_std_w` | Rolling std, window w | ✅ |
| `res_roll_var_w` | Rolling variance | ✅ |
| `res_roll_median_w` | Rolling median | ✅ |
| `res_roll_min_w`, `res_roll_max_w` | Rolling extrema | ✅ |
| `res_roll_abs_mean_w` | Mean absolute residual | ✅ |
| `res_ewm_mean` | EWMA of residuals | ✅ |
| `res_ewm_var` | EWM variance | ✅ |
| `res_energy_w` | Mean squared residual over window w | ✅ |
| `res_slope_w` | Linear slope over last w steps | ✅ |
| `res_skew_w`, `res_kurt_w` | Rolling skewness / kurtosis | ✅ |
| `res_acf_hat_lag96` | Rolling lag-96 autocorrelation estimate | ✅ |
| `res_sign_persistence` | Fraction of same-sign steps in window | ✅ |

> **Note:** These are mathematically computable from the 96-step residual vector the GRU already receives. They may still help optimization (explicit inductive bias) but do **not** add information capacity.

---

## C. Prophet / level context features

| Feature | Definition | Online? |
|---------|------------|---------|
| `prophet_yhat` | Prophet in-sample / forecast level at t | ✅ |
| `prophet_trend` | Prophet trend component | ✅ |
| `prophet_daily` | Prophet daily seasonal component | ✅ |
| `cpu_scaled` | Actual scaled CPU at t | ✅ (past window only) |
| `prophet_uncertainty` | Prophet interval width (if enabled) | ✅ |
| `level_bin` | Quantile bin of prophet_yhat on train | ✅ |

**Key identity:** For past steps, `cpu_scaled[t] = prophet_yhat[t] + residual[t]`. Adding level + residual is equivalent to exposing CPU level alongside residual dynamics.

---

## D. Container static context (train-computed constants)

| Feature | Definition | Online? |
|---------|------------|---------|
| `cpu_mean` | Train mean scaled CPU | ✅ |
| `cpu_std` | Train std scaled CPU | ✅ |
| `cpu_max`, `cpu_min` | Train extrema | ✅ |
| `cpu_cv` | Coefficient of variation | ✅ |
| `container_volatility_tier` | Stable / medium / spiky (EDA label) | ✅ |
| `train_residual_std` | Std of Prophet train residuals | ✅ |
| `train_ljung_box_stat` | Container residual structure score | ✅ (offline precompute) |

---

## E. Regime / peak indicators

| Feature | Definition | Online? |
|---------|------------|---------|
| `is_peak_p90` | 1 if cpu_scaled ≥ train P90 | ✅ |
| `is_peak_p95` | Stricter peak flag | ✅ |
| `peak_distance` | Steps since last peak | ✅ |
| `near_peak` | Within k steps of peak | ✅ |
| `regime_high` | Above train median CPU | ✅ |
| `change_point_flag` | Recent mean shift detector | ⚠️ (noisy at 15 min) |

---

## F. CPU-side rolling context (not in current Hybrid)

| Feature | Definition | Online? |
|---------|------------|---------|
| `cpu_roll_mean_w` | Rolling mean of past cpu_scaled | ✅ |
| `cpu_roll_std_w` | Rolling std of past cpu_scaled | ✅ |
| `cpu_ewm` | EWMA of cpu_scaled | ✅ |
| `cpu_diff1` | First difference of CPU | ✅ |

Partially overlaps Prophet + residual decomposition.

---

## G. Cross-series / multivariate (Alibaba raw columns)

| Feature | Definition | Online? |
|---------|------------|---------|
| `mem_util_percent` | Memory utilization | ❌ ~98% missing |
| `net_in`, `net_out` | Network I/O | ❌ ~98% missing |
| `disk_io_percent` | Disk utilization | ❌ ~98% missing |
| `cpi`, `mpki`, `mem_gps` | Hardware counters | ❌ ~99% missing |
| `machine_id` | Host identity | ❌ sparse |

---

## H. Additional statistically meaningful features

| Feature | Rationale |
|---------|-----------|
| `res_zscore_local` | Residual divided by rolling std (local standardized shock) |
| `prophet_yhat_z` | Prophet level relative to container train mean/std |
| `interaction_peak_x_vol` | Peak flag × rolling residual std |
| `res_abs` | Absolute residual magnitude |
| `sign(residual)` | Direction of correction |
| `forecast_horizon_phase` | Position within 96-step day (for future work on horizon-aware models) |

---

## Feature count summary

| Category | Count | Practical for Hybrid v2 |
|----------|-------|-------------------------|
| Calendar | 10 | 0–2 (likely redundant) |
| Residual transforms | 20+ | 1–2 (volatility only; rest redundant) |
| Prophet / level | 6 | 1–2 |
| Container static | 7 | 1 |
| Regime / peak | 6 | 1 |
| CPU rolling | 4 | 0–1 |
| Multivariate | 6 | 0 |

**Realistic compact set:** 4–6 channels total including residual.
