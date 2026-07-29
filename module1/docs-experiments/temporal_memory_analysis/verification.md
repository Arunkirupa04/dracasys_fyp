# Verification

## Authoritative run

`experiments/temporal_memory_analysis_2026-07-26_170052/`

## Data integrity

| Check | Result |
|-------|--------|
| Containers expected | 99 |
| Containers analyzed | 99 |
| All containers in inference cache | Yes |
| CPU full timeline steps (sample) | 768 |
| Effective max lag (CPU) | 672 |
| Hybrid series length | 96 |
| Model training in this experiment | None |
| Prophet.fit() in this experiment | None |

Source: `verification/data_integrity.json`

## Dataset hashes (frozen config)

SHA-256 hashes recorded for:

- `data/train_df.parquet`
- `data/val_df.parquet`
- `data/scalers.pkl`
- `data/selected_containers.npy`
- `experiments/peak_aware_2026-07-14_164518/evaluation/baseline_inference_cache.pkl`

## Reproducibility

```bash
MPLCONFIGDIR=.matplotlib_cache .venv/bin/python scripts/run_temporal_memory_analysis.py
```

Fixed bootstrap seed: **12345** (2000 resamples, 95% CI).

## Read-only audit

| Prior experiment | Modified |
|------------------|----------|
| Hybrid baseline | No |
| Global GRU | No |
| Peak-Aware | No |
| CSRLE | No |
| LFHE | No |
| RRA | No |
| RRA Audit | No |

## Output completeness

| Directory | Expected content | Status |
|-----------|------------------|--------|
| `config/` | Frozen JSON | ✓ |
| `acf/`, `pacf/` | Matrices + cohort summaries | ✓ |
| `fft/` | Per-series CSV summaries | ✓ |
| `statistics/` | Daily, memory, window, paired JSON | ✓ |
| `plots/` | PNG + PDF figures | ✓ |
| `tables/` | Cohort summary CSV | ✓ |
| `reports/` | final_report.md + .json | ✓ |
| `verification/` | data_integrity.json | ✓ |

## Notebook

`notebooks/temporal_memory_analysis.ipynb` — manual execution; points to authoritative run timestamp.

Build:

```bash
.venv/bin/python scripts/build_temporal_memory_analysis_notebook.py
```
