# Ridge Residual Analysis Audit — Report

**Timestamp:** 2026-07-26_162500  
**Audited:** `experiments/ridge_residual_analysis_2026-07-26_161500/` (read-only)  
**Verdict:** RRA v1.0 remaining-residual conclusions **invalidated by scaling bug**

## Executive summary

The counterintuitive ACF increase (0.199 → 0.266) and Ljung–Box increase (75.8% → 96.0%) in RRA v1.0 are **not scientifically valid**. They were caused by subtracting **z-scored Ridge predictions** from **raw Prophet residuals** without inverse scaling.

After correction (all operations in z-space):

- Mean |ACF| remaining: **0.201** (Δ = +0.001 vs Prophet)
- Ljung–Box reject remaining: **73.7%** (Δ = −2.0 pp vs Prophet)

## Task summaries

### Task 1 — Residual definition
- Sign correct: `remaining = prophet − ridge`
- **Scaling bug confirmed** for all 99 containers

### Task 2 — Alignment
- Day-1 block aligned; no off-by-one in first 96 val steps

### Task 3 — Data splits
- No leakage; Ridge trained on train only

### Task 4 — ACF
- RRA CSV replicated; corrected metrics differ materially from RRA remaining

### Task 5 — Why ACF appeared to increase
1. Ridge explains only ~6% variance (r ≈ 0.11)
2. Raw − z subtraction inflates variance (~3.3× vs ~1.1× corrected)
3. Bogus remainder has distorted autocorrelation

### Task 6 — Linearity
- AR(1) on corrected remaining: forecast r ≈ **0.38** (similar to Prophet AR(1) ≈ 0.39)
- Remaining structure is **predominantly linear**

### Task 7 — Decision

1. **Ridge failed as predictor** — not a successful linear structure remover  
2. **Remaining dependence is linear** — AR(1) adequate  
3. **Ridge→GRU NOT justified** — additional linear modelling more appropriate

## Recommendation

Do **not** proceed to Ridge→GRU based on RRA v1.0. If residual modelling continues, first fix scaling, improve linear Ridge performance, and demonstrate AR models fail before considering nonlinear stages.
