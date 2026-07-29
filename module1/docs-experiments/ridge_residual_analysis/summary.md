# RRA Summary

## Experiment

**Ridge Residual Analysis v1.0** — diagnostic on CSRLE B0-LC, 99 containers. Trains Ridge only.

## Verdict

**`structure_persists_minimal_ridge_gain`**

## One-sentence takeaway

Linear Ridge (96→96, α=1.0) **does not whiten** Prophet residuals; remaining series show **more** autocorrelation and **96%** white-noise rejection — so temporal structure persists and a **future** Ridge→GRU architecture experiment remains scientifically justified, though Ridge alone adds little beyond Prophet.

## Explicit answers

| # | Question | Answer |
|---|----------|--------|
| 1 | How much structure did Ridge remove? | **Negative reduction** — \|ACF\| increased by ~0.067 |
| 2 | Ridge residual closer to white noise? | **No** |
| 3 | % containers still reject white noise? | **96.0%** |
| 4 | Significant temporal dependence remains? | **Yes** |
| 5 | Nonlinear learner justified? | **Yes** |
| 6 | Ridge→GRU in future experiment? | **Yes** (dedicated protocol required) |

## Artifacts

- Report: `experiments/ridge_residual_analysis_2026-07-26_161500/reports/final_report.md`
- Notebook: `notebooks/ridge_residual_analysis.ipynb`
- Docs: `docs/ridge_residual_analysis/`

## Thesis use

> Before stacking a GRU after Ridge, we verified that Ridge regression on Prophet residuals does not eliminate temporal dependence. Remaining residuals reject white-noise tests in 96% of containers and exhibit higher short-lag |ACF| than Prophet residuals alone, motivating a controlled Ridge→GRU architecture experiment rather than assuming the linear stage suffices.
