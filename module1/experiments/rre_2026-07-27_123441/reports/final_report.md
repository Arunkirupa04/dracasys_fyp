# Residual Scaling Experiment (RRE v1.0)

**Protocol:** rre_v1.0
**Timestamp:** 2026-07-27_123441

## Research question

Can a more optimisation-friendly scaling of the SAME Prophet residual improve Hybrid GRU learning without changing the underlying temporal information?

## Stage 0 context

Stage 0 rejected velocity (R1) because it removed temporal structure. R3 preserves the same ACF/PACF as R0 while improving numerical robustness (lower sparsity in scaled units). This experiment tests whether that improves GRU optimisation — not whether a better temporal representation exists.

## Cohort summaries

- R0 mean day1 MAE: 1.7398671431611448
- R3 mean day1 MAE: 1.7450872015883665

## Research question answers

### q1_accuracy_improved
- **Answer:** False
- **Evidence:** Mean day1 MAE delta (R3-R0)=0.0052; fraction improved=45.45%

### q2_optimisation_improved
- **Answer:** False
- **Evidence:** R3 better on 3/8 optimisation metrics; best_epoch R0=5 R3=4

### q3_convergence_speed
- **Answer:** True
- **Evidence:** Best epoch R0=5, R3=4

### q4_generalisation
- **Answer:** False
- **Evidence:** Generalisation gap R0=0.8555, R3=6.3016

### q5_statistically_significant
- **Answer:** False
- **Evidence:** Wilcoxon p=0.9749404863818756; Pearson Wilcoxon sig=False

### q6_practically_meaningful
- **Answer:** False
- **Evidence:** Mean MAE improvement=-0.0052 (threshold 0.05)

### q7_replace_baseline_scaling
- **Answer:** False
- **Evidence:** Requires accuracy improvement that is both statistically and practically significant; otherwise retain R0 global z-score as Hybrid standard.

### q8_optimisation_vs_temporal_information
- **Answer:** Stage 0 showed R3 preserves identical temporal structure to R0. Any difference in this experiment reflects optimisation/numerical conditioning, not new temporal information.
- **Evidence:** Residual Pearson R0 mean=0.04727203088857314, R3 mean=0.0502228529572965

## Statistical test (day1 MAE, R3 vs R0)

- Mean delta: 0.005220058427222165
- Wilcoxon p: 0.9749404863818756
- Bootstrap 95% CI: {'n_pairs': 99, 'mean_diff': 0.005220058427222165, 'median_diff': 0.0002818944466220774, 'ci_low': -0.0021560074298971337, 'ci_high': 0.015450958695129129, 'fraction_left_better': 0.5454545454545454}
