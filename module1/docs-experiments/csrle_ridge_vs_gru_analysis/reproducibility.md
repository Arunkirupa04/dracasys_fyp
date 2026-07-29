# Reproducibility

## Environment

```bash
cd /Users/lehaan/Desktop/FYP_Long_Term
source tf_metal_env/bin/activate   # optional
```

Dependencies: pandas, numpy, scipy, statsmodels, scikit-learn, matplotlib, prophet, tensorflow (GRU load only).

## Batch analysis (writes plots + CSVs)

```bash
./tf_metal_env/bin/python3 experiments/csrle_ridge_vs_gru_analysis_2026-07-26_013200/run_analysis.py
```

## Interactive notebook

```bash
jupyter notebook notebooks/csrle_ridge_vs_gru_analysis.ipynb
```

Run all cells sequentially from repo root (notebook uses `Path('..')` for repo root).

## Regenerate notebook JSON

```bash
./tf_metal_env/bin/python3 experiments/csrle_ridge_vs_gru_analysis_2026-07-26_013200/build_notebook.py
```

## Verification

Check `experiments/.../verification/artifact_verification.csv` — all `exists=True`.

## Frozen paths

Configured in `experiments/.../config/paths.json`.
