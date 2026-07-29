# Task 10 — Feature Ranking

Ranked by **expected usefulness**, **scientific justification**, **implementation complexity**, and **risk** (lower risk = better).

Scoring: 1 (low) – 5 (high) for usefulness and justification; complexity/risk inverted so 5 = easiest/lowest risk.

---

## Ranked candidates

| Rank | Feature | Usefulness | Justification | Impl. | Risk | Notes |
|------|---------|:----------:|:-------------:|:-----:|:----:|-------|
| **1** | `prophet_yhat_scaled` | 5 | 5 | 4 | 4 | Level on seasonal surface; not encoded in residual alone; peak-aware + RPA show level-dependent errors |
| **2** | `res_roll_std_12` (causal) | 4 | 4 | 5 | 4 | Explicit heteroscedasticity; LFHE links dispersion failure to objective; complements shape learning |
| **3** | `is_peak_p90` (train threshold) | 4 | 4 | 5 | 3 | Regime nonlinearity; peak-aware pipeline already defines thresholds; some overlap with level |
| **4** | `cpu_std` (container broadcast) | 3 | 4 | 5 | 5 | Global GRU uses it; static scale prior; zero time-varying correlation but modulates correction amplitude |
| **5** | `hour_sin`, `hour_cos` | 2 | 2 | 5 | 5 | Cheap ablation for Prophet mis-specification; TMA suggests low prior |
| 6 | `cpu_mean` (broadcast) | 2 | 3 | 5 | 5 | Partially redundant with prophet level |
| 7 | `train_residual_std` | 2 | 3 | 5 | 5 | Correction-scale prior; overlaps cpu_std |
| 8 | `res_lag96` | 2 | 2 | 5 | 4 | TMA: lag-96 ACF ≈ 0.04 — likely redundant |
| 9 | `res_energy_96` | 2 | 2 | 5 | 4 | Redundant with roll_std |
| 10 | `cpu_scaled` (past) | 4* | 3 | 5 | 3 | *High usefulness but collinear with prophet_yhat + residual; pick one level encoding |
| 11 | `res_roll_mean_*` | 2 | 1 | 5 | 3 | Deterministic from residual window |
| 12 | `res_diff1`, `res_slope_12` | 2 | 1 | 5 | 3 | Same — GRU-internalizable |
| 13 | `weekday`, `day_index` | 1 | 1 | 5 | 5 | Prophet + short span |
| 14 | `mem_util`, `net_*`, `disk_io` | 1 | 1 | 1 | 2 | Data not available |

---

## Top 5 recommendation (for future experiment)

1. **`residual_scaled`** — retain (baseline channel)
2. **`prophet_yhat_scaled`** — primary new context
3. **`res_roll_std_12`** — local volatility
4. **`cpu_std`** — container dispersion prior
5. **`is_peak_p90`** — regime flag

**Optional ablation swap:** replace `is_peak_p90` with `hour_sin/cos` to test calendar gap hypothesis at negligible cost.

---

## Decision matrix

| If priority is… | Choose |
|-----------------|--------|
| Maximum new information | `prophet_yhat_scaled` + `is_peak_p90` |
| LFHE / dispersion angle | `res_roll_std_12` + `cpu_std` |
| Minimum complexity | `prophet_yhat_scaled` only (+1 channel) |
| Safest ablation | Add one feature at a time to 96×2, then 96×3 |

---

## Features explicitly **not** recommended

- Full rolling feature suite (redundant, overfitting)
- Both `cpu_scaled` and `prophet_yhat` simultaneously
- Both `res_roll_std` and `res_ewm_var`
- Multivariate Alibaba columns
- `container_id` embedding (insufficient data per ID for 100-container cohort)
