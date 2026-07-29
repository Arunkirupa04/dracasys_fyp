# Limitations

## What TMA does not claim

TMA is a **read-only diagnostic on temporal autocorrelation structure**. The following claims are **explicitly outside its scope**. No TMA document or report should be read as making these claims:

| TMA does NOT claim | Explanation |
|--------------------|-------------|
| **Longer windows improve MAE** | No models were retrained or evaluated under alternative window lengths. |
| **Longer windows improve RMSE** | Same — no forecast comparison was conducted. |
| **Hybrid should use longer windows** | Hybrid GRU receives Prophet residuals (lag-96 ACF ≈ 0.04), not raw CPU. Residual memory is short. |
| **Global GRU accuracy would necessarily improve** | Autocorrelation at long lags does not guarantee that a GRU can exploit it for better forecasts. |
| **GRU failed because of insufficient context** | CSRLE demonstrated GRU can learn synthetic structure; LFHE identified dispersion issues; RRA Audit found remaining structure is predominantly linear. |

**What TMA does claim:** It quantifies how much temporal information (autocorrelation energy) is present at different window lengths in each of three distinct signals. Whether a model can convert that information into forecast accuracy is a separate, untested question.

---

## Read-only constraints

1. **No Prophet.fit()** — Prophet-equivalent residuals use train-fitted Fourier daily seasonality + linear trend (OLS), not actual Prophet in-sample fits. Residual magnitudes may differ slightly from Hybrid training residuals.

2. **Hybrid post-forecast residual length** — Limited to frozen validation **day-1** (96 steps) from `baseline_inference_cache.pkl`. Long-lag analysis at daily multiples is not defined for this series.

3. **Temporal span vs validation scope** — Cohort is validation-evaluable containers; long-lag CPU/Prophet analysis uses **full train+val timeline** (~768 steps) because validation-only (~154 steps) cannot support 672-lag ACF.

## Metric limitations

4. **Integrated autocorrelation time** — Can be near zero or negative when ACF alternates (daily periodicity). It understates memory when strong periodic peaks exist at lags 96, 192, …

5. **Hypothetical window mean |ACF|** — Averaging |ACF| over 1..W decreases as W grows even when cumulative capture increases. Window sufficiency (cumulative squared-ACF) is the primary sufficiency metric.

6. **Cumulative squared-ACF capture** — Heuristic energy proxy; not equivalent to forecastable information or model R².

## General

7. **No forecast evaluation** — This experiment does not retrain models or compare MAE/RMSE under longer windows.

8. **Single trace length** — ~7–8 days per container after preprocessing limits maximum lag to 672 and may not represent longer-horizon cloud workloads.

9. **Cohort size** — 99 containers; bootstrap CIs assume container-level independence.

10. **Architecture-specific interpretation required** — Results from original CPU (Signal 1) must not be applied to Hybrid GRU without noting that Hybrid operates on Prophet residuals (Signal 2).

## No modification of prior experiments

Hybrid, Global GRU, Peak-Aware, CSRLE, LFHE, RRA, and Audit artifacts were read only.

## Documentation-only updates

Interpretation sections in this documentation set were expanded after the initial TMA run to clarify Global vs Hybrid implications and research narrative positioning. **Numerical results were not changed.**
