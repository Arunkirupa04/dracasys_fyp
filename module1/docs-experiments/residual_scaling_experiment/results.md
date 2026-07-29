# Results

**Run:** `experiments/rre_2026-07-27_123441`  
**Cohort:** 99 evaluable containers  
**Protocol:** `rre_v1.0`

## Forecast accuracy (primary)

| Metric | R0 (global z-score) | R3 (median/MAD) | Δ (R3−R0) | Wilcoxon p |
|--------|---------------------|-----------------|-----------|------------|
| Day-1 MAE (%) | **1.740** | 1.745 | +0.0052 | 0.975 |
| Day-1 RMSE (%) | **2.380** | 2.385 | +0.0050 | 0.161 |
| Residual Pearson r | 0.047 | 0.050 | +0.003 | 0.709 |
| Residual std ratio | 0.038 | **0.054** | +0.016 | <0.001 |

R3 does **not** improve Day-1 MAE or RMSE. Differences are negligible and not statistically significant for primary CPU metrics.

## Optimisation behaviour

| Metric | R0 | R3 |
|--------|----|----|
| Best epoch | 5 | **4** |
| Epochs until early stop | 15 | **14** |
| Best val MSE (scaled space) | 1.71 | 12.55* |
| Generalisation gap (val−train at best) | 0.86 | 6.30* |

\*Raw MSE values are **not directly comparable** across arms because R3 scaled residuals have ~7× higher variance (Stage 0). Convergence epoch counts are comparable; loss magnitudes are not.

R3 reached best epoch one step earlier, but this did not translate into better forecasts or lower generalisation gap in a scale-normalised sense.

## Statistical significance

- **Day-1 MAE:** not significant (Wilcoxon p = 0.975; bootstrap 95% CI includes 0)
- **Day-1 RMSE:** not significant (p = 0.161)
- **Peak MAE:** Wilcoxon p = 0.002 (R3 slightly better), but bootstrap 95% CI includes 0 — not practically meaningful (~0.008% mean improvement)
- **Residual std ratio:** significant (R3 predicts higher residual variance relative to actual) — reflects scaling artefact, not CPU accuracy

## Practical significance

Mean MAE change = **−0.0052%** (R3 worse), well below the 0.05% threshold. **Not practically meaningful.**

## Research question answers

| # | Question | Answer |
|---|----------|--------|
| 1 | Accuracy improved? | **No** |
| 2 | Optimisation improved? | **No** (marginal epoch gain only) |
| 3 | Convergence speed? | **Marginally** (best epoch 4 vs 5) |
| 4 | Generalisation? | **No** |
| 5 | Statistically significant? | **No** (primary metrics) |
| 6 | Practically meaningful? | **No** |
| 7 | Replace baseline scaling? | **No — retain R0 global z-score** |
| 8 | Optimisation vs temporal info? | Confirms Stage 0: same temporal content; scaling change did not unlock better learning |

## Conclusion

**Negative/null result.** Robust median/MAD scaling does not improve Hybrid forecasting accuracy or optimisation outcomes in a meaningful way. The frozen global z-score baseline remains the scientifically justified standard.
