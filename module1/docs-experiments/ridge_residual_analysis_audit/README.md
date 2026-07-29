# Ridge Residual Analysis Audit (RRAA)

## Status

**COMPLETE** — `experiments/ridge_residual_analysis_audit_2026-07-26_162500/`

## Purpose

Read-only audit of RRA v1.0 (`experiments/ridge_residual_analysis_2026-07-26_161500/`) after counterintuitive ACF increase (0.199 → 0.266) and Ljung–Box rejection increase (75.8% → 96.0%).

## Verdict

**RRA v1.0 remaining-residual metrics are invalidated by a scaling bug.** After correction, ACF and Ljung–Box barely change. Remaining structure is **predominantly linear**. **Ridge→GRU is not scientifically justified** on current evidence; additional linear modelling is more appropriate.

## Key finding

RRA subtracted **z-scored Ridge predictions** from **raw Prophet validation residuals** without inverse scaling — and extended history with mixed spaces. This artifactually inflated |ACF| and Ljung–Box rejections.

## Authoritative artifacts

| Path | Content |
|------|---------|
| `reports/audit_report.json` | Full audit outcomes |
| `verification/task1_scaling_mismatch.csv` | Per-container scaling evidence |
| `tables/linear_model_ar_summary.csv` | AR(1/2/5) on corrected remaining |
| `notebooks/ridge_residual_analysis_audit.ipynb` | Interactive audit notebook |

## Audited experiment (read-only)

`experiments/ridge_residual_analysis_2026-07-26_161500/` — **not modified**

## Documentation

| Document | Purpose |
|----------|---------|
| [methodology.md](methodology.md) | Audit protocol |
| [verification.md](verification.md) | Tasks 1–4 checks |
| [results.md](results.md) | Quantitative outcomes |
| [discussion.md](discussion.md) | Interpretation & decisions |
| [summary.md](summary.md) | Thesis-ready summary |

## Re-run

```bash
python scripts/run_ridge_residual_analysis_audit.py
python scripts/build_ridge_residual_analysis_audit_notebook.py
```
