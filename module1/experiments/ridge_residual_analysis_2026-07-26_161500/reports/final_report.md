# Ridge Residual Analysis — Final Report

**Timestamp:** `2026-07-26_161500`  
**Condition:** b0_lc  
**Containers:** 99  
**Verdict:** `structure_persists_minimal_ridge_gain`

## Research question

Does Ridge remove all predictable temporal structure from Prophet residuals,
or does the remaining Ridge residual still justify a nonlinear learner (GRU)?

## Ridge training

- Alpha: 1.0
- Sequence: 96→96
- Internal split: chronological_80_20

## Cohort Ridge performance (validation)

- Mean Pearson r: 0.1251
- Mean MAE (scaled): 0.1405
- Mean remaining std ratio: 1.8183

## Temporal structure comparison

| Metric | Prophet residual | Ridge remaining | Reduction |
|--------|------------------|-----------------|-----------|
| Mean \|ACF\| lags 1–10 | 0.1995 | 0.2660 | -0.0666 |
| Ljung–Box reject fraction (lag 20) | 75.8% | 96.0% | -20.2 pp |

## Explicit answers

1. **How much temporal structure did Ridge remove?**  
   Mean |ACF| reduction = -0.0666; mean |PACF| reduction = -0.0312.

2. **Is the Ridge residual closer to white noise?**  
   No (remaining |ACF| = 0.2660 vs Prophet 0.1995).

3. **What percentage of containers still reject white noise?**  
   96.0% (Ljung–Box lag 20, p<0.05).

4. **Does the Ridge residual still contain statistically significant temporal dependence?**  
   Yes (cohort mean |ACF| lags 1–10 = 0.2660).

5. **Is there sufficient evidence to justify a nonlinear learner after Ridge?**  
   Yes.

6. **Should a Ridge→GRU architecture be implemented in a future experiment?**  
   Yes — pending dedicated architecture experiment.

## Case studies

- Best Ridge: `c_15179`
- Median: `c_10032`
- Worst Ridge: `c_12231`

## Conclusion

Temporal structure persists after Ridge with limited ACF/Ljung reduction. A nonlinear learner may still be justified, but Ridge adds little beyond Prophet in this linear 96→96 configuration.