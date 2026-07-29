# RRA Results

**Authoritative run:** `experiments/ridge_residual_analysis_2026-07-26_161500/`

## Cohort Ridge performance (validation)

| Metric | Value |
|--------|-------|
| Mean Pearson r | 0.125 |
| Median Pearson r | 0.126 |
| Mean MAE (scaled) | 0.141 |
| Mean remaining std ratio | 1.818 |
| Mean var reduction fraction | −8.08 (variance increased) |

Ridge explains only ~12% linear alignment with Prophet residuals on validation. Subtraction yields a **higher-variance** remainder.

## Temporal structure comparison

| Metric | Prophet residual | Ridge remaining | Δ (Prophet − Remaining) |
|--------|------------------|-----------------|-------------------------|
| Mean \|ACF\| lags 1–10 | 0.199 | 0.266 | **−0.067** |
| Mean \|PACF\| lags 1–10 | 0.109 | 0.140 | **−0.031** |
| Ljung–Box reject fraction (lag 20) | 75.8% | 96.0% | **−20.2 pp** |

Bootstrap 95% CI for ACF reduction: **[−0.091, −0.041]** — significantly **negative** (remaining has **more** ACF, not less).

## White-noise proximity

| Question | Result |
|----------|--------|
| Ridge residual closer to white noise? | **No** |
| Containers rejecting white noise (remaining, lag 20) | **96.0%** |
| Containers rejecting white noise (Prophet, lag 20) | 75.8% |

## Explicit answers (from final report)

1. **Temporal structure removed by Ridge?** Minimal / negative — |ACF| and Ljung rejections **increased** after subtraction.
2. **Ridge residual closer to white noise?** **No.**
3. **% containers still rejecting white noise?** **96.0%.**
4. **Significant temporal dependence in remaining?** **Yes** (mean |ACF| 1–10 = 0.266).
5. **Nonlinear learner justified?** **Yes** — structure clearly persists.
6. **Ridge→GRU architecture in future experiment?** **Yes** — but Ridge alone is insufficient; any stack must address why linear 96→96 fails to whiten.

## Case studies

| Role | Container | Notes |
|------|-----------|-------|
| Best Ridge | `c_15179` | Highest validation Pearson r |
| Median | `c_10032` | Closest to cohort median r |
| Worst Ridge | `c_12231` | Lowest validation Pearson r |

Plots: `plots/case_study_{best,median,worst}_*.png`

## Verdict

**`structure_persists_minimal_ridge_gain`**
