# LFHE Summary

## Experiment

**LFHE v1.1** — MSE vs DA-MSE (λ=1.0) on CSRLE B0-LC, 99 containers.

## Verdict

**`inconclusive_I1`** — dispersion recovered, correlation not, MAE guardrail failed.

## Explicit answers

| # | Question | Answer |
|---|----------|--------|
| 1 | Did DA-MSE reduce variance collapse? | **Yes** (std ratio +0.73) |
| 2 | Did prediction variance increase? | **Yes** |
| 3 | Did energy recovery improve? | **Yes** (ERR 0.01→0.82) |
| 4 | Did Pearson r improve? | **No** (0.113→0.087) |
| 5 | Did CPU forecasting improve? | **No** |
| 6 | Hypothesis supported? | **No** |
| 7 | Variance collapse explanation | Objective contributes to dispersion; λ=1.0 DA-MSE over-corrects without shape gain |
| 8 | Thesis conclusion | Partial objective effect; not sufficient for Hybrid residual shape learning |

## One-sentence takeaway

Adding dispersion supervision at λ=1.0 **fixes under-dispersion too aggressively** — confirming MSE permits collapsed variance, but **rejecting** simple objective swap as a complete solution.

## Artifacts

- Report: `experiments/loss_function_hypothesis_2026-07-26_130252/reports/final_report.md`
- Notebook: `notebooks/lfhe_visualization.ipynb`
- Design: `docs/loss_function_experiment/`
