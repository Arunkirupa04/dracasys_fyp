# RRAA Results

**Audit run:** `experiments/ridge_residual_analysis_audit_2026-07-26_162500/`

## Scaling bug impact

| Quantity | RRA v1.0 (buggy) | Corrected z-space |
|----------|------------------|-------------------|
| Mean \|ACF\| 1–10 remaining | 0.266 | 0.201 |
| Δ vs Prophet | +0.067 | +0.001 |
| Ljung reject remaining | 96.0% | 73.7% |
| Δ vs Prophet | +20.2 pp | −2.0 pp |
| Mean var ratio (remaining/prophet) | ~3.3 (std ratio²) | 1.10 |

## Ridge predictor quality (corrected z-space)

| Metric | Value |
|--------|-------|
| Cohort mean Pearson r | 0.105 |
| Cohort mean R² | 0.059 |
| Verdict | Ridge explains ~6% variance — **failed as predictor** |

## Task 6 — Linear models on corrected remaining

Walk-forward AR one-step forecasts (cohort mean):

| Series | AR order | Forecast r | Forecast MAE | AR-resid \|ACF\| 1–10 | AR-resid Ljung reject |
|--------|----------|------------|--------------|----------------------|----------------------|
| prophet_z | 1 | 0.393 | 0.773 | 0.100 | 27.3% |
| prophet_z | 2 | 0.386 | 1.215 | 0.095 | 25.3% |
| **remaining_corr_z** | **1** | **0.380** | **0.700** | **0.092** | **24.2%** |
| remaining_corr_z | 2 | 0.377 | 0.732 | 0.088 | 24.2% |
| remaining_corr_z | 5 | 0.348 | 1.455 | 0.087 | 21.2% |

AR(1) on corrected remaining achieves **similar** forecast r to AR(1) on Prophet residuals. Remaining structure is **still linearly predictable** at AR(1) level.

## Task 7 — Explicit answers

1. **Ridge failed to remove linear dependence, or failed as predictor?**  
   **Both.** Primary issue: **predictor failure** (r ≈ 0.11). After scaling correction, Ridge **does not** materially reduce |ACF| (−0.001).

2. **Remaining dependence linear or nonlinear?**  
   **Predominantly linear.** AR(1) forecast r ≈ 0.38 on corrected remaining; AR residuals still reject white noise in only ~24% of containers.

3. **Ridge→GRU justified vs more linear modelling?**  
   **Additional linear modelling is more appropriate.** Ridge→GRU is **not** scientifically justified on measured evidence.
