# Executive Summary — Context Feature Feasibility

**Study type:** Design and feasibility only (no training, no new experiments).  
**Date:** 2026-07-26

---

## Question

Should Hybrid GRU move from **96×1** (residual only) to **96×N** (residual + context), and is the required information available in the Alibaba dataset without leakage?

---

## Short answer

| # | Question | Answer |
|---|----------|--------|
| 1 | Enough data for context-enriched Hybrid? | **Yes** — CPU, Prophet level, train stats, peak flags, causal volatility. **No** usable multivariate telemetry. |
| 2 | Genuinely new vs Prophet-duplicative? | **New:** Prophet level, local volatility, container scale, peak regime. **Duplicative:** hour/day, lag-96, most rolling means, residual lags/diffs. |
| 3 | Top 3–5 features for future v2? | (1) `prophet_yhat_scaled`, (2) `res_roll_std_12`, (3) `cpu_std`, (4) `is_peak_p90`, (5) optional `hour_sin/cos` |
| 4 | Scientifically justified to investigate? | **Yes, conditionally** — plausible representational gap, consistent with TMA/LFHE/RPA; **weak prior for large gains** per RLLA |

---

## Key evidence integration

```
RPA        → residuals structured after Prophet
CSRLE      → GRU CAN learn when structure is strong (synthetic)
LFHE       → variance fix ≠ correlation fix; MSE matters
TMA        → Prophet removes daily memory; 96-step OK for Hybrid; calendar redundant
RLLA       → real residuals weakly predictable (r ≈ 0.02–0.06)
Peak-aware → regime thresholds exist; level matters at peaks
```

**Consistency:** Hybrid failure is **not** primarily insufficient input **length** (TMA). It may partly reflect **missing level/regime context** in a **weak-signal** environment where MSE further suppresses dispersion (LFHE).

---

## Proposed Hybrid v2 input

```
Shape: (96, 5)

Channels: residual_scaled | prophet_yhat_scaled | res_roll_std_12 | cpu_std | is_peak_p90
```

See [hybrid_v2_design.md](hybrid_v2_design.md).

---

## Read-only analysis highlights

From 99-container validation-region correlation (Prophet-equivalent residuals):

| Feature | Mean \|r\| with residual | Interpretation |
|---------|--------------------------|----------------|
| `cpu_scaled` | 0.771 | Strong level coupling — not in residual-only GRU |
| `is_peak_p90` | 0.626 | Regime dependence |
| `res_roll_std_12` | 0.155 | Volatility coupling |
| `hour` | 0.017 | Prophet absorbed calendar |

Figures: `figures/context_feature_residual_correlation.png`, `figures/acf_cpu_vs_prophet_residual_example.png`

---

## Verdict

**Investigate Hybrid v2 with a compact 4–5 channel context tensor.** Do not implement wide feature expansion. Pair any future training experiment with residual Pearson r and std ratio metrics, and consider joint loss-function study if dispersion collapse persists.

---

## Explicit final answers

### 1. Does our current dataset contain enough information to build a context-enriched Hybrid model?

**Yes, for CPU-centric context** (Prophet level, rolling residual volatility, container train statistics, peak indicators derived from train thresholds). The dataset does **not** provide usable 15-minute multivariate telemetry (memory, network, disk are ~98%+ missing).

### 2. Which contextual features are genuinely new information, rather than duplicates of what Prophet already models?

**Genuinely new (recommended):**
- `prophet_yhat_scaled` — operating level on the fitted seasonal surface
- `res_roll_std_12` — local heteroscedasticity of corrections
- `cpu_std` — container-specific dispersion prior
- `is_peak_p90` — high-load regime indicator

**Duplicates / low value:**
- `hour`, `minute_of_day`, `day_index`, `weekday` (Prophet daily seasonality + TMA lag-96 ≈ 0.04)
- `res_lag96`, rolling means, diffs, slopes (deterministic from 96-step residual input)
- `mem_util`, `net_in/out`, `disk_io`, hardware counters (missing data)

### 3. Which 3–5 features would you recommend for a future Hybrid v2 experiment?

1. **`prophet_yhat_scaled`**
2. **`res_roll_std_12`**
3. **`cpu_std`** (container broadcast)
4. **`is_peak_p90`**
5. **`hour_sin` / `hour_cos`** (optional cheap ablation — low prior)

Baseline channel **`residual_scaled`** retained as channel 0.

### 4. Based on all evidence collected in this research, is a context-enriched Hybrid model scientifically justified?

**Yes — as a focused hypothesis worth testing, not as a guaranteed improvement.**

Justification: RPA and CSRLE show residual structure exists and is learnable under favourable conditions; TMA shows the fix is **not** longer calendar memory but possibly **level/regime conditioning**; LFHE shows shape learning decouples from dispersion; RLLA bounds expected gains on real data. A compact context-enriched Hybrid v2 is **consistent with the full research chain** and **feasible without leakage**, but **large MAE improvements should not be expected** without complementary changes (e.g., loss function).
