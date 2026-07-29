# RRA Verification

## Data integrity

| Check | Status |
|-------|--------|
| B0-LC train parquet SHA256 | Matches LFHE frozen config |
| B0-LC val parquet SHA256 | Matches LFHE frozen config |
| Container count | 99 evaluable |
| CSRLE artifacts modified | **No** |
| LFHE artifacts modified | **No** |
| Hybrid / Global GRU modified | **No** |

See `verification/data_integrity.json`.

## Training boundary

| Model | Trained in RRA? |
|-------|----------------|
| Ridge (diagnostic) | **Yes** |
| GRU | **No** |
| Prophet | Fit per-container at analysis time (same as Hybrid pipeline; not persisted as experiment artifact) |

## Reproducibility

```bash
# From repository root (requires sklearn, statsmodels, prophet, pandas, matplotlib)
python scripts/run_ridge_residual_analysis.py 2026-07-26_161500
```

Expected verdict: `structure_persists_minimal_ridge_gain` (deterministic Ridge; Prophet/cmdstan may vary slightly by platform).

## Notebook

`notebooks/ridge_residual_analysis.ipynb` — manual execution; points to `experiments/ridge_residual_analysis_2026-07-26_161500/`.

Update `EXP_DIR` in Section 1 if re-run with a new timestamp.
