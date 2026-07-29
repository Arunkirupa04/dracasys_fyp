# Methodology

## Scope

Read-only analysis of **frozen CSRLE Stage 2 B0** artifacts. GRU weights loaded from `stage2_synthetic_gru/b0_lc/models/`. Ridge predictions reproduced using the **identical Stage 2 protocol** (96→96, α=1.0, train-sequence 80% chronology) — not a new hyperparameter search.

## Data

- **B0** synthetic train/val parquets (frozen CSRLE v2)  
- **Stage 2** extended metrics and baseline comparison CSVs  
- **α map** and injection manifest (generator properties)  
- **Condition A / C0** pooled residuals for complexity comparison only  

## Residual definition

Day-1 Prophet residual in scaled CPU space:

`r(t) = cpu_scaled(t) − prophet_yhat(t)`

Same as CSRLE Stage 2 `extended_metrics` and residual-pattern analysis Section 13.

## Analyses

| # | Module | Methods |
|---|--------|---------|
| 1 | Linearity | ACF, PACF, Welch/FFT spectrum, AR(1)/AR(2)/polynomial R² |
| 2 | Complexity | Permutation entropy, sample entropy (A vs B0 vs C0) |
| 3 | Frequency | Welch spectra; dominant frequency; spectral correlation |
| 4 | Variance | Predicted vs actual std; std ratio by model |
| 5 | Trajectories | Actual vs GRU vs Ridge time series |
| 6 | Error decomposition | Bias, amplitude ratio, Pearson r, phase lag (cross-correlation) |
| 7 | Horizons | Metrics at h ∈ {1,4,12,24,48,96} |
| 8 | Containers | Top/bottom ranks; scatter GRU r vs Ridge r |
| 9 | Generator covariates | Correlation α, κ_eff with Ridge−GRU gap |
| 10 | Synthesis | Cohort aggregates integrated into JSON summary |

## Representative container selection

Container closest to cohort **median** of `(ridge_r − gru_r)` — systematic, not cherry-picked for visual appeal.

## Outputs

All new files under `experiments/csrle_ridge_vs_gru_analysis_2026-07-26_013200/`.
