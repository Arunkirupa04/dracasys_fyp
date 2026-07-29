# Implementation Log

## Module structure

```
utils/hcerl/
  config.py          — variants, frozen hyperparameters
  features.py        — leakage-safe feature engineering
  training.py        — train_hcerl_variant, artifact I/O
  inference.py       — multi-channel Day-1 inference
  evaluation.py      — extended metrics + cache
  ablation.py        — paired deltas, bootstrap, Wilcoxon
  plots.py           — publication PNG/PDF
  case_studies.py    — automatic case selection
  report.py          — final_report.md/json
```

## Runner

`scripts/run_hcerl_experiment.py` — trains V0–V4 sequentially, evaluates, ablates, plots, reports.

**Environment:** Use `tf_metal_env/bin/python` (TensorFlow required for GRU training).

**Authoritative run:** `experiments/hcerl_2026-07-26_180649/`

**Post-processing only:** `scripts/finish_hcerl_experiment.py experiments/hcerl_2026-07-26_180649`

## Experiment output layout

```
experiments/hcerl_{timestamp}/
  config/hcerl_config_frozen.json
  variants/{v0_baseline,...,v4_full}/
    models/
    training/
    evaluation/
    verification/
  ablation/
  plots/
  reports/
  case_studies/
```

## Isolation

No edits to `utils/hybrid_*.py`, baseline experiment dirs, or prior study artifacts.

## Seed policy

Matches frozen `train_hybrid_gru` — no TF seed set. Documented in config for cross-variant fairness note.
