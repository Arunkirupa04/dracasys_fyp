# Ridge Residual Analysis (RRA)

## Status

**COMPLETE** — `experiments/ridge_residual_analysis_2026-07-26_161500/`

## Purpose

Read-only diagnostic: after Ridge regression on Prophet residuals (CSRLE B0-LC), does **remaining temporal structure** still justify a nonlinear learner (GRU)?

This experiment does **not** implement Ridge→GRU. It provides scientific evidence for or against a future architecture experiment.

## Verdict

**`structure_persists_minimal_ridge_gain`**

Ridge (α=1.0, 96→96) achieves modest prediction quality (mean r ≈ 0.13) but **does not whiten** Prophet residuals. Remaining series show **higher** |ACF| and **more** Ljung–Box rejections than Prophet residuals alone — structure persists and a nonlinear follow-up remains scientifically justified.

## Authoritative artifacts

| Path | Content |
|------|---------|
| `config/ridge_residual_analysis_config.json` | Frozen protocol rra_v1.0 |
| `ridge/models/ridge_model.pkl` | Diagnostic Ridge model |
| `tables/ridge_evaluation_per_container.csv` | Per-container Ridge metrics |
| `diagnostics/temporal_structure_per_container.csv` | ACF/PACF/Ljung/FFT |
| `diagnostics/white_noise_comparison.json` | Cohort white-noise comparison |
| `reports/final_report.json` | Machine-readable outcomes |
| `notebooks/ridge_residual_analysis.ipynb` | Interactive dashboard (manual run) |

## Key results (99 containers, B0-LC validation)

| Metric | Prophet residual | Ridge remaining |
|--------|------------------|-----------------|
| Mean \|ACF\| lags 1–10 | 0.199 | **0.266** |
| Ljung–Box reject (lag 20) | 75.8% | **96.0%** |
| Mean Pearson r (Ridge pred) | — | 0.125 |

## Documentation index

| Document | Purpose |
|----------|---------|
| [methodology.md](methodology.md) | Protocol and pipeline |
| [implementation_log.md](implementation_log.md) | Build chronology |
| [diagnostics.md](diagnostics.md) | Diagnostic definitions |
| [results.md](results.md) | Quantitative outcomes |
| [discussion.md](discussion.md) | Interpretation |
| [verification.md](verification.md) | Integrity checks |
| [summary.md](summary.md) | Thesis-ready summary |

## Re-run

```bash
python scripts/run_ridge_residual_analysis.py
python scripts/build_ridge_residual_analysis_notebook.py
```

Uses CSRLE B0-LC frozen data only. No GRU training. No modification of prior experiments.
