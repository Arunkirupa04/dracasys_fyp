# LFHE Final Report

**Protocol:** lfhe_v1.1
**Timestamp:** 2026-07-26_130252

## Reproduction gate (MSE control vs Stage 2 B0)
- **Pass:** True

## Cohort metrics

| Metric | MSE | DA-MSE |
|--------|-----|--------|
| Pearson r | 0.1133 | 0.0874 |
| Std ratio | 0.1007 | 0.8268 |
| Residual MAE | 0.0932 | 0.1047 |
| Energy recovery | 0.0110 | 0.8232 |

## Hypothesis verdict
- **Verdict:** `inconclusive_I1`
- S1 dispersion: True
- S2 correlation: False
- S3 MAE guardrail: False

## Explicit answers

1. Did DA-MSE reduce variance collapse? **True**
2. Did prediction variance increase? **True**
3. Did energy recovery improve? **True**
4. Did Pearson correlation improve? **False**
5. Did CPU forecasting improve? **False**
6. Was the hypothesis supported? **False**
7. Variance collapse explanation: Partial/inconclusive: inconclusive_I1
8. Thesis conclusion code: `inconclusive_I1`
