# LFHE Reproducibility

## Authoritative run

`experiments/loss_function_hypothesis_2026-07-26_130252/`

## Re-run full experiment

```bash
cd /path/to/FYP_Long_Term
python scripts/run_lfhe_experiment.py
```

Creates a **new timestamped directory** — does not overwrite prior runs.

## Re-run notebook

```bash
python scripts/build_lfhe_visualization_notebook.py
jupyter notebook notebooks/lfhe_visualization.ipynb
```

Update `LFHE_EXP` path in notebook cell 1 to target the desired run.

## Frozen inputs (read-only)

- `experiments/synthetic_residual_learnability_2026-07-25_164307/data/b0_lc/`
- Stage 2 B0 model (reproduction gate only)

## Seeds

| Component | Value |
|-----------|-------|
| GRU training | None (CSRLE parity) |
| Bootstrap | 12345 |

## Environment

`tf_metal_env` — TensorFlow with Metal on Apple Silicon.

Record `pip freeze` in experiment `logs/` if auditing.

## Dataset hashes

In `config/lfhe_config_frozen.json` → `dataset_hashes`.
