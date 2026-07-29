# Success Criteria

## Statistical significance

For primary comparison vs G96:

| Criterion | Threshold |
|-----------|-----------|
| Bootstrap 95% CI on Δ MAE | Entire interval < 0 → significant improvement |
| Wilcoxon p-value | < 0.05 |
| Paired containers | n = 99 (same evaluable cohort) |

Report **both** bootstrap and Wilcoxon — either alone is insufficient for thesis claim.

## Effect size (practical significance)

| Metric | Small | Medium | Large |
|--------|-------|--------|-------|
| Cohen's d on Δ MAE | 0.2 | 0.5 | 0.8 |

**Minimum for "meaningful":** |d| ≥ 0.2 **and** cohort mean |Δ MAE| ≥ **1.0% relative** (≈ 0.017 pp on ~1.7 MAE scale).

## Primary success (H1 supported)

All of:

1. Cohort mean Day-1 MAE improves vs G96
2. Bootstrap CI excludes zero (improvement direction)
3. ≥ 55% containers improved (MAE)
4. |Cohen's d| ≥ 0.2

## Secondary success (subgroup story)

1. Significant improvement in **high-memory tertile** (TMA-derived)
2. Correlation |ρ| ≥ 0.25 between TMA capture gap and per-container Δ MAE
3. Low-memory tertile shows no significant degradation

## Parsimony success (recommended deployment)

Shortest window `W*` such that:

- MAE within **+0.01 pp** of best variant
- Not statistically worse than best (bootstrap CI for Δ MAE vs best includes zero)

## Failure / inconclusive criteria

| Verdict | Condition |
|---------|-----------|
| **Reject H1** | CI includes zero; |d| < 0.2 |
| **Inconclusive** | MAE improves but CI includes zero (underpowered or high variance) |
| **Reject long windows** | G288 significantly **worse** than G96 |

## Metrics beyond MAE

Report but do not use alone for verdict:

- RMSE (scale-sensitive peaks)
- MAPE (noisy at low CPU — same caveats as baseline)
- Training/validation loss curves (overfitting diagnostic)

## GGTCE does NOT require beating Hybrid

Success is **within Global architecture** — beating Hybrid is a separate thesis comparison already covered by baseline phases.

## Documentation success

Protocol satisfied if:

- All variants logged with sequence counts
- TMA subgroup analysis completed
- Explicit answer to all five final feasibility questions in `summary.md`
