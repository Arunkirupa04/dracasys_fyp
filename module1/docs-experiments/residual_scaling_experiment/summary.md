# Summary

**Experiment:** RRE v1.0 — Residual Scaling Experiment  
**Question:** Does robust median/MAD scaling improve Hybrid GRU learning without changing temporal information?  
**Arms:** R0 (global z-score, frozen baseline) vs R3 (median/MAD)

## Why this experiment exists

Stage 0 showed velocity removes temporal structure and R3 preserves the same ACF as R0. Training-side interventions (HCERL, peak-aware, windows) failed. The remaining testable hypothesis is **optimisation conditioning**, not new temporal representations.

## What was held constant

Prophet, GRU architecture, sequences (96→96), Adam+MSE, epochs/early stopping, cohort (99 containers), evaluation metrics, bootstrap/Wilcoxon protocol.

## What changed

Only residual scaling: `(r - μ)/σ` vs `(r - median)/MAD`, with matching inverse at inference.

## Deliverables

| Item | Location |
|------|----------|
| Code | `utils/rre/` |
| Runner | `scripts/run_rre_experiment.py` |
| Notebook | `notebooks/residual_scaling_experiment.ipynb` |
| Docs | `docs/residual_scaling_experiment/` |
| Artifacts | `experiments/rre_<timestamp>/` |

## Eight research questions

Answered in `reports/final_report.json` after experiment run:

1. Does robust scaling improve accuracy?
2. Does it improve optimisation behaviour?
3. Convergence speed?
4. Generalisation?
5. Statistical significance?
6. Practical meaningfulness?
7. Replace baseline scaling?
8. Optimisation vs temporal information?

**Negative results are valid** and should be reported as evidence that global z-score remains the appropriate Hybrid standard.

**Full arc (Stage 0 + RRE):** [docs/hybrid_residual_investigation/CONCLUSION.md](../hybrid_residual_investigation/CONCLUSION.md)
