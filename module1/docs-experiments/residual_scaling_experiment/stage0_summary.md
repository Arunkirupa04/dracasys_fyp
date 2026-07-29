# Stage 0 Summary

Stage 0 (`rre_diagnostics_v1.0`) was a **read-only** statistical study comparing R0–R3 before any GRU training.

## Conclusions relevant to RRE v1.0

| Representation | Temporal structure | Decision |
|----------------|-------------------|----------|
| R0 Level z-score | Strongest mean \|ACF\| | Baseline |
| R1 Velocity | Integrated τ ≈ 0 | **Rejected** |
| R2 Per-container z-score | Same ACF as R0; worse kurtosis | Not selected |
| R3 Median/MAD | **Identical ACF to R0**; lowest sparsity | Primary challenger |

## Why velocity was not tested in RRE v1.0

R1 reduced mean |ACF| from 0.035 to 0.012 and collapsed integrated autocorrelation time. Stage 0 demonstrated it **removes** temporal information rather than exposing it. Including R1 in a GRU experiment would confound optimisation effects with representation loss.

## Why R3 was selected

R3 preserves the same temporal dependence structure as R0 (affine scaling does not change autocorrelation) while:

- Reducing fraction of near-zero scaled values (3.3% vs 10.5% at |x| < 0.05)
- Down-weighting extreme outliers via MAD < σ for skewed residuals
- Potentially improving gradient conditioning for MSE + Adam

## What Stage 0 did NOT answer

Stage 0 could not determine whether robust scaling improves **GRU optimisation or forecast accuracy**. That requires training — which is the purpose of RRE v1.0.

## Reference artifacts

- Config: `experiments/rre_diagnostics_2026-07-27_120759/config/rre_diagnostics_config.json`
- Comparison: `experiments/rre_diagnostics_2026-07-27_120759/comparison/comparison_table.csv`
- Findings: `docs/residual_representation_diagnostics/FINDINGS.md`
