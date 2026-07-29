# Tasks 3–6 — Feature Analysis

For each feature class: (1) computable from existing data?, (2) online at inference?, (3) leakage?, (4) duplicates Prophet?, (5) new information for GRU?, (6) computational cost.

Analysis uses read-only correlation on 99-container cohort (`context_feature_residual_correlation.csv`) and Temporal Memory Analysis (TMA) run `2026-07-26_170052`.

---

## Task 3 — Per-feature evaluation matrix

### Temporal / calendar

| Feature | (1) Data? | (2) Online? | (3) Leakage? | (4) Prophet dup? | (5) New info? | (6) Cost |
|---------|-----------|-------------|--------------|------------------|---------------|----------|
| `hour` / cyclical | ✅ | ✅ | No | **High** — daily seasonality | **Low** — mean \|r\| ≈ 0.017 | Trivial |
| `minute_of_day` | ✅ | ✅ | No | High | Low | Trivial |
| `day_index` | ✅ | ✅ | No | Partial — trend | Low | Trivial |
| `weekday` / `is_weekend` | ✅ | ✅ | No | High — weekly off | **Very low** — weekly_seasonality=False, short span | Trivial |
| `step_index` | ✅ | ✅ | No | Partial — trend | Low | Trivial |

### Residual transforms

| Feature | (1) | (2) | (3) | (4) | (5) | (6) |
|---------|-----|-----|-----|-----|-----|-----|
| `res_lag1`, `res_diff1` | ✅ | ✅ | No | No | **No** — in 96-step window | Trivial |
| `res_roll_mean_w` | ✅ | ✅ | No | No | **No** — linear function of window | O(w) |
| `res_roll_std_12` | ✅ | ✅ | No | No | **Marginal** — heteroscedasticity cue; \|r\| ≈ 0.16 | O(w) |
| `res_roll_std_96` | ✅ | ✅ | No | No | Marginal | O(w) |
| `res_ewm_var_12` | ✅ | ✅ | No | No | Marginal — corr with roll_std ≈ 0.94 | O(1) incremental |
| `res_energy_96` | ✅ | ✅ | No | No | Low — \|r\| ≈ 0.07 | O(w) |
| `res_lag96` | ✅ | ✅ | No | Partial — daily | **Low** — TMA lag-96 ACF ≈ 0.04 | Trivial |
| `res_slope_12` | ✅ | ✅ | No | No | No — derivable | O(w) |
| `res_skew_w` | ✅ | ✅ | No | No | Low | O(w) |

### Prophet / level

| Feature | (1) | (2) | (3) | (4) | (5) | (6) |
|---------|-----|-----|-----|-----|-----|-----|
| `prophet_yhat` | ✅ | ✅ | No if in-sample past / forecast for known future ts | Partial — defines residual | **Yes** — level not in residual alone | Prophet predict (already computed) |
| `cpu_scaled` (past window) | ✅ | ✅ | No for past window | Partial | **Yes** — algebraically adds level | Trivial |
| `prophet_daily` component | ✅ | ✅ | No | **High** — explicit seasonality | Low | From Prophet components |
| `prophet_trend` | ✅ | ✅ | No | High | Low | From Prophet components |

### Container static

| Feature | (1) | (2) | (3) | (4) | (5) | (6) |
|---------|-----|-----|-----|-----|-----|-----|
| `cpu_std` | ✅ | ✅ | No — train-only | No | **Yes** — global scale prior | Trivial (broadcast) |
| `cpu_mean` | ✅ | ✅ | No | Partial — level shift | Moderate | Trivial |
| `train_residual_std` | ✅ | ✅ | No | No | Moderate — correction scale | One-time precompute |

### Regime / peak

| Feature | (1) | (2) | (3) | (4) | (5) | (6) |
|---------|-----|-----|-----|-----|-----|-----|
| `is_peak_p90` | ✅ | ✅ | No — threshold from train | No | **Yes** — nonlinear regime; \|r\| ≈ 0.63* | Trivial |
| `peak_distance` | ✅ | ✅ | No | No | Moderate | O(window) |

\* High correlation partly reflects level-residual coupling at peaks; still a distinct **regime flag** not explicit in residual-only input.

### Multivariate

| Feature | (1) | (2) | (3) | (4) | (5) | (6) |
|---------|-----|-----|-----|-----|-----|-----|
| `mem_util_percent` | ⚠️ 1.6% | ❌ | — | — | Unknown | — |
| All other raw telemetry | ❌ | ❌ | — | — | — | — |

---

## Task 4 — Temporal features vs Prophet + TMA

### What Prophet models

Hybrid Prophet config: `daily_seasonality=True`, `weekly_seasonality=False`, fit per container on train.

Prophet explicitly estimates:
- **Trend** (piecewise linear growth)
- **Daily seasonal Fourier terms** (96-step period at 15-min resolution)

### TMA evidence on residuals

From `experiments/temporal_memory_analysis_2026-07-26_170052`:

| Signal | Lag-96 mean ACF | Interpretation |
|--------|-----------------|----------------|
| Original CPU | **0.33** | Strong daily memory |
| Prophet-equivalent residual | **0.042** | Daily cycle largely removed |
| Hybrid post-forecast residual | ~3-step memory | Short-range error correlation |

Daily-cycle PSD fraction on Prophet-equivalent residuals ≈ **0.7%** (vs dominant for raw CPU).

### Conclusion on calendar features

Adding `hour`, `minute_of_day`, or `day_index` to the GRU is **likely redundant** with Prophet's daily seasonality component because:

1. TMA shows **93%+ reduction** in lag-96 autocorrelation after Prophet-equivalent decomposition.
2. Empirical residual–hour correlation is **near zero** (cohort mean r ≈ 0.017).
3. Temporal Memory Analysis supports the **96-step residual window** — the missing signal is not "longer calendar context" but **short-range residual dynamics and level-dependent effects**.

**Exception (low cost ablation):** cyclical `hour_sin/cos` as a 1-channel probe for Prophet **seasonality mis-specification** (container-specific phase errors). Prior is low; cost is negligible.

### Weekday / weekend

Train span ≈ 6.5 days; weekly seasonality disabled. **Not justified** as a primary feature.

---

## Task 5 — Statistical-context features

### Rolling mean / EWMA of residuals

- **Information:** local level of correction series; smooths short-run bias.
- **Help residual prediction?** Only if GRU fails to internalize running mean — CSRLE shows GRU *can* learn structure when signal is strong; on real data RLLA shows weak linear predictability.
- **Complements vs duplicates:** **Duplicates** residual history (linear functional).

### Rolling std / variance / volatility

- **Information:** local heteroscedasticity — "how large are corrections recently?"
- **Help?** **Plausible** — LFHE showed variance collapse is a major failure mode; explicit volatility channel may encourage state-dependent correction scale even when MSE penalizes dispersion.
- **Complements vs duplicates:** Partially derivable from window, but **nonlinear squaring** may be harder for GRU to extract under MSE; **recommended as explicit channel**.

### Rolling energy (mean squared residual)

- Correlates with variance; **redundant with rolling std** (see Task 6).

### Rolling skew / kurtosis

- RPA: pooled residual kurtosis ≈ 58 (heavy tails). Skew/kurtosis capture tail asymmetry.
- **Marginal** — high variance estimates at w=12; **not first priority**.

### Recent slope / differences

- Encode velocity of correction; **duplicate** of lag structure GRU should learn.
- RLLA: AR(1) r ≈ 0.02 on real data — short-lag linear signal is weak.

### Peak / regime indicators

- Capture **nonlinear** response at high CPU — not reducible to residual history alone without also knowing level.
- **Recommended** when paired with train-derived threshold (peak-aware pipeline already computes P90).

---

## Task 6 — Feature correlation and compact set

### High inter-feature redundancy (|r| > 0.85)

| Pair | r | Action |
|------|---|--------|
| `hour` ↔ `minute_of_day` | 0.999 | Keep at most one cyclical pair |
| `res_roll_std_12` ↔ `res_ewm_var_12` | 0.942 | Keep **one** volatility proxy |
| `res_roll_mean` ↔ `res_diff1` ↔ `res_lag1` | high | **Drop** — redundant with raw residual window |
| `cpu_scaled` ↔ `is_peak` | partial | Peak is thresholded CPU; prefer **one** level/regime encoding |
| `cpu_scaled` ↔ `prophet_yhat` | not identical | With residual fixed, adding both CPU and prophet is over-parameterized; pick **one level feature** |

### Residual-derived vs context (conceptual)

Correlation with residual **does not imply incremental GRU value**:
- `res_roll_mean`, `res_diff1` show \|r\| ≈ 0.5–0.6 with residual because they **are** the residual series transformed.
- The GRU already receives 96 consecutive residuals — these channels add **capacity**, not **information**.

### Recommended compact feature set (5 channels)

| # | Channel | Role |
|---|---------|------|
| 0 | `residual_scaled` | Primary signal (current) |
| 1 | `prophet_yhat_scaled`* | Operating level on seasonal surface |
| 2 | `res_roll_std_12` | Local volatility / heteroscedasticity |
| 3 | `cpu_std` (broadcast) | Container scale prior |
| 4 | `is_peak_p90` | High-load regime indicator |

\*Scale `prophet_yhat` with train-only stats (container or global) for numerical stability.

**Optional 6th (ablation only):** `hour_sin`, `hour_cos` — test Prophet seasonality gap.

**Drop from consideration:** multivariate telemetry, rolling means of residual, lag96, energy96, weekday, step_index, duplicate volatility measures.

### Correlation summary (validation region, 99 containers)

| Feature | Mean Pearson r with residual |
|---------|------------------------------|
| `cpu_scaled` | 0.771 |
| `is_peak_p90` | 0.626 |
| `res_roll_std_12` | 0.155 |
| `res_lag96` | 0.118 |
| `prophet_pred` | −0.071 |
| `res_energy_96` | 0.067 |
| `hour` | 0.017 |
| `cpu_std` (pooled) | ~0 (constant within container) |

Interpretation: high r for `cpu_scaled` / `is_peak` confirms **level-regime coupling** absent from residual-only input. Near-zero r for `hour` confirms Prophet absorbed calendar structure.
