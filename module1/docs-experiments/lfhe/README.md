# LFHE — Loss Function Hypothesis Experiment

## Status

**COMPLETE** — `experiments/loss_function_hypothesis_2026-07-26_130252/`

## Purpose

Prospective test: does **Dispersion-Augmented MSE (DA-MSE)** reduce variance collapse vs standard MSE under identical Hybrid architecture, Prophet, and CSRLE B0 data?

## Verdict

**`inconclusive_I1`** — dispersion and energy recovery improved substantially, but Pearson r did not improve and MAE guardrail was violated.

## Authoritative artifacts

| Path | Content |
|------|---------|
| `config/lfhe_config_frozen.json` | Frozen protocol v1.1 |
| `b0_lc_mse/` | MSE control arm (LFHE session retrain) |
| `b0_lc_da_mse/` | DA-MSE treatment arm |
| `reports/final_report.json` | Machine-readable outcomes |
| `notebooks/lfhe_visualization.ipynb` | Interactive dashboard |
| `notebooks/lfhe_mse_vs_damse_comparison.ipynb` | MSE vs DA-MSE thesis comparison |

## Design protocol

Frozen design: [docs/loss_function_experiment/](../loss_function_experiment/)

## Key results (cohort mean)

| Metric | MSE (LFHE) | DA-MSE |
|--------|------------|--------|
| Pearson r | 0.113 | 0.087 |
| Std ratio | 0.101 | **0.827** |
| Residual MAE | 0.093 | 0.105 |
| Energy recovery | 0.011 | **0.823** |

Stage 2 B0 reference (frozen weights): r=0.084, std ratio=0.073 — reproduction gate **PASSED**.

## Documentation index

| Document | Purpose |
|----------|---------|
| [implementation_log.md](implementation_log.md) | Build chronology |
| [training.md](training.md) | Training outcomes |
| [evaluation.md](evaluation.md) | Primary metrics |
| [diagnostic_results.md](diagnostic_results.md) | Horizon + energy diagnostics |
| [findings.md](findings.md) | Key findings |
| [mse_vs_damse_comparison_analysis.md](mse_vs_damse_comparison_analysis.md) | Full comparison notebook analysis |
| [discussion.md](discussion.md) | Interpretation |
| [verification.md](verification.md) | Integrity checks |
| [reproducibility.md](reproducibility.md) | Re-run commands |
| [summary.md](summary.md) | Thesis-ready summary |
