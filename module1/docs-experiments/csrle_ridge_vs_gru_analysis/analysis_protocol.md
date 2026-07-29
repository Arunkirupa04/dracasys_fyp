# Analysis Protocol

Each analysis maps to notebook sections and `run_analysis.py` modules.

## Analysis 1 — Residual linearity

**Question:** Are B0 validation residuals mostly linearly predictable after Prophet?

**Procedure:**
1. Pool Day-1 B0 validation residuals across 99 containers  
2. Compute ACF/PACF (lags 0–40)  
3. Welch power spectrum  
4. Fit AR(1), AR(2), linear and quadratic trend models; report R²  

**Interpretation threshold:** High short-lag ACF and AR R² support linear forecastability.

## Analysis 2 — Complexity

**Question:** Is B0 residual complexity unexpectedly low vs A and C0?

**Metrics:** Permutation entropy (order 3), sample entropy (m=2, r=0.2·σ)

## Analysis 3 — Frequency domain

**Question:** Does Ridge preserve spectral content better than GRU?

**Per container:** Welch spectra for actual, GRU, Ridge; dominant frequency error; normalized spectral correlation.

## Analysis 4 — Variance recovery

**Question:** Does GRU collapse toward low-variance predictions?

**Metrics:** `std(pred)/std(actual)` cohort distribution; scatter actual std vs predicted std.

## Analysis 5 — Trajectories

Full 96-step and zoomed 1–24 step overlays on identical axes.

## Analysis 6 — Error decomposition

| Component | Definition |
|-----------|------------|
| Bias | mean(pred − actual) |
| Amplitude | std(pred)/std(actual) |
| Correlation | Pearson r |
| Phase lag | argmax cross-correlation lag |

## Analysis 7 — Horizons

Within-trajectory MAE, r, std ratio at h = 1, 4, 12, 24, 48, 96; cohort mean.

## Analysis 8 — Container behaviour

Rank containers by GRU r and Ridge r; identify systematic Ridge advantage.

## Analysis 9 — Generator properties

Correlate `ridge_r − gru_r` with α, κ_eff, actual residual std.

## Analysis 10 — Evidence synthesis

Integrate cohort means into `results/evidence_synthesis.json`; no unsupported claims.
