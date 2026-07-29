# Implementation Log

## 2026-07-26 — Analysis project created

| Item | Detail |
|------|--------|
| **Directory** | `experiments/csrle_ridge_vs_gru_analysis_2026-07-26_013200/` |
| **Notebook** | `notebooks/csrle_ridge_vs_gru_analysis.ipynb` |
| **Docs** | `docs/csrle_ridge_vs_gru_analysis/` |
| **Code** | `analysis_lib.py`, `run_analysis.py`, `build_notebook.py` (new analysis only) |
| **Frozen inputs** | CSRLE Stage 2 B0 model, stats, evaluation CSVs, synthetic parquets |
| **Constraint** | No modification to prior experiments |

## Execution

```bash
./tf_metal_env/bin/python3 experiments/csrle_ridge_vs_gru_analysis_2026-07-26_013200/run_analysis.py
./tf_metal_env/bin/python3 experiments/csrle_ridge_vs_gru_analysis_2026-07-26_013200/build_notebook.py
```

Log: `experiments/.../analysis_run.log`

## 2026-07-26 — Analysis execution complete

| Output | Status |
|--------|--------|
| `results/evidence_synthesis.json` | Written |
| 11 CSV result tables | Written |
| 14 plot pairs (PNG+PDF) | Written |
| `verification/artifact_verification.csv` | All artifacts exist |

Key synthesis: GRU r=0.084, Ridge r=0.158; GRU std ratio=0.073, Ridge=0.220; B0 AR(1) R²=0.201.
