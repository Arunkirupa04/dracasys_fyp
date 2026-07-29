# Residual Scaling Experiment (RRE v1.0)

**Protocol:** `rre_v1.0`  
**Type:** Isolated research experiment — R0 (global z-score) vs R3 (median/MAD)

## Research question

Can a more optimisation-friendly scaling of the **same** Prophet residual improve Hybrid GRU learning without changing the underlying temporal information?

## Critical distinction

| Stage 0 (diagnostics) | RRE v1.0 (this experiment) |
|----------------------|----------------------------|
| Compares representations statistically | Trains two Hybrid GRU models |
| No GRU training | Only scaling differs |
| Rejected velocity (R1) | Tests R0 vs R3 only |
| Showed R3 = same ACF as R0 | Tests optimisation, not new temporal info |

## Variants

| ID | Scaling | Role |
|----|---------|------|
| **R0** | `(r - μ) / σ` global train | Frozen Hybrid baseline |
| **R3** | `(r - median) / MAD` global train | Robust scaling challenger |

Everything else is frozen: GRU architecture, Prophet, sequences, optimizer, loss, epochs, early stopping, cohort, metrics.

## Layout

```
utils/rre/
scripts/run_rre_experiment.py
scripts/build_rre_notebook.py
notebooks/residual_scaling_experiment.ipynb
docs/residual_scaling_experiment/
experiments/rre_<timestamp>/
```

## Run

```bash
tf_metal_env/bin/python scripts/run_rre_experiment.py
tf_metal_env/bin/python scripts/build_rre_notebook.py
```

Update notebook `EXP_DIR` to the generated timestamp folder.

## Isolation

Does **not** modify Hybrid baseline, Global GRU, production pipeline, Stage 0 diagnostics, or prior experiments.

## Documentation

See `docs/residual_scaling_experiment/` for full thesis-oriented documentation.

**Consolidated conclusion (Stage 0 + RRE v1.0):** [docs/hybrid_residual_investigation/CONCLUSION.md](../hybrid_residual_investigation/CONCLUSION.md)
