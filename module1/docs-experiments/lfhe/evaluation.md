# LFHE Evaluation

## Protocol

Identical to CSRLE Stage 2: Day-1 96-step holdout, 99 containers, scaled residual metrics + CPU metrics.

## Cohort summary (LFHE session retrain)

| Metric | MSE | DA-MSE | Δ (DA−MSE) |
|--------|-----|--------|------------|
| Pearson r | 0.1133 | 0.0874 | −0.0259 |
| Std ratio | 0.1007 | 0.8268 | +0.7261 |
| Residual MAE | 0.0932 | 0.1047 | +0.0114 |
| Residual RMSE | 0.1379 | 0.1500 | +0.0121 |
| R² | −0.331 | −0.456 | — |

## Paired bootstrap (95% CI)

| Metric | Mean Δ | CI | frac DA better |
|--------|--------|-----|----------------|
| Pearson r | −0.026 | [−0.055, 0.003] | 43% |
| Std ratio | +0.726 | [0.529, 0.988] | **100%** |
| MAE | +0.011 | [0.010, 0.013] | 10% (DA worse) |

## CPU metrics

Hybrid − Prophet MAE delta negligible for both arms (consistent with CSRLE).

## Hypothesis criteria (frozen)

| Criterion | Result |
|-----------|--------|
| S1 Dispersion | **PASS** (std ratio 0.827 ≥ 0.12; CI excludes 0) |
| S2 Correlation | **FAIL** (r decreased; CI includes 0) |
| S3 MAE guardrail | **FAIL** (Δ MAE 0.0114 > 0.010) |

**Verdict:** `inconclusive_I1`

## Artifacts

- `b0_lc_mse/evaluation/extended_metrics.csv`
- `b0_lc_da_mse/evaluation/extended_metrics.csv`
- `comparisons/paired_mse_vs_da_mse.json`
- `tables/lfhe_summary.csv`
