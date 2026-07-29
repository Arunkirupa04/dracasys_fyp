# RRAA Discussion

## Root cause of counterintuitive RRA result

RRA v1.0 applied Ridge in **z-scored** space (global train mean/std) but subtracted predictions from **raw** Prophet validation residuals without inverse scaling:

```python
# RRA v1.0 (bug)
remaining = prophet_raw - ridge_pred_z

# Correct (Hybrid / CSRLE convention)
remaining = prophet_raw - (ridge_pred_z * res_std + res_mean)
# equivalently: remaining_z = prophet_z - ridge_pred_z
```

Additionally, when extending the input history across validation blocks, RRA concatenated z-scored train tail with **raw** validation residuals — mixing scales in the Ridge input window.

### Why this inflated |ACF|

Z-scored Ridge predictions have std ≈ 0.10 while raw Prophet residuals have std ≈ 0.11 — similar magnitude but **not commensurate** after subtraction. The bogus remainder `raw − z` has inflated variance (cohort var ratio ~3.3 vs ~1.1 corrected) and distorted autocorrelation structure, producing artificially higher |ACF| and Ljung–Box rejections.

This is **not** evidence that Ridge increases temporal dependence — it is a **measurement artifact**.

## After correction

| Observation | Interpretation |
|-------------|----------------|
| \|ACF\| 0.199 → 0.201 | Ridge subtraction barely changes autocorrelation |
| Ljung 75.8% → 73.7% | Slight **decrease** in white-noise rejection |
| Ridge r ≈ 0.11 | Linear 96→96 model captures little structure |
| AR(1) r ≈ 0.38 on remaining | What structure remains is **linearly** forecastable |

## Implications for RRA v1.0 conclusions

| RRA claim | Audit verdict |
|-----------|---------------|
| ACF increased after Ridge | **Invalid** — scaling bug |
| 96% reject white noise | **Invalid** — drops to 73.7% corrected |
| Nonlinear learner justified | **Not supported** — remaining is AR(1)-predictable |
| Ridge→GRU future experiment | **Not justified** on current evidence |

## What would be needed before Ridge→GRU

1. Fix scaling in any future diagnostic (z-space throughout, or inverse-scale before subtraction).
2. Demonstrate Ridge removes meaningful linear structure (|ACF| reduction, lower Ljung reject) — **not observed** with α=1.0 Ridge.
3. Show AR(p) models **fail** on corrected remaining while \|ACF\| stays high — **not observed** (AR(1) works similarly to Prophet).
4. Only then would nonlinear justification have empirical basis.

## Relation to CSRLE / LFHE

CSRLE Stage 2 already showed Ridge > GRU on B0 correlation. This audit explains **why stacking GRU after Ridge is premature**: Ridge barely predicts, does not whiten, and remaining structure is linear — not a nonlinear residual niche.
