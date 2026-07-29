# Implementation

## Module structure

```
utils/rre/
├── config.py       # Frozen protocol, variants, metrics
├── scaling.py      # ScalingStats, R0/R3 fit and transform
├── training.py     # train_rre_variant()
├── inference.py    # run_rre_inference()
├── evaluation.py   # Cohort evaluation + optimisation summary
├── comparison.py   # Paired bootstrap + Wilcoxon
└── report.py       # Research question answers
```

## Key design decisions

1. **Reuse `build_hybrid_gru_model` and `generate_prophet_residuals`** via import — no modification of `utils/hybrid_training.py`.
2. **Return `HybridInferenceResult`** from inference — enables reuse of `compute_container_extended_metrics()` without modification.
3. **Isolation namespace** — all RRE logic under `utils/rre/`; artifacts under `experiments/rre_<timestamp>/`.
4. **Stage 0 untouched** — scaling formulas mirror Stage 0 definitions but are implemented independently in `utils/rre/scaling.py`.

## Training flow

```python
train_residual_df = generate_prophet_residuals(global_train)
stats = fit_scaling_stats(train_residual_df, variant_id)  # R0 or R3
enriched_df = apply_scaling(train_residual_df, stats)
# → create_residual_sequences → train GRU (frozen protocol)
```

## Inference flow

```python
train_container["residual_scaled"] = scaling_stats.scale(residual)
# GRU predict → scaling_stats.inverse() → add Prophet → inverse MinMax
```

## Artifact layout

```
experiments/rre_<timestamp>/
├── config/rre_config_frozen.json
├── variants/R0/
│   ├── models/{hybrid_gru.keras, scaling_stats.pkl}
│   ├── training/{training_history.csv, optimization_summary.json}
│   └── evaluation/{evaluation_extended.csv, cohort_summary.json}
├── variants/R3/  (same structure)
├── comparison/{paired_tests.json, per_container_paired.csv}
└── reports/{final_report.json, final_report.md}
```

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_rre_experiment.py` | Train R0 + R3, evaluate, compare |
| `scripts/build_rre_notebook.py` | Generate thesis notebook |

## Runtime

Prophet fitting on 99 containers × 2 variants ≈ 40–50 minutes on Apple Silicon with `tf_metal_env`.
