# Reproducibility

## Experiment identity

| Field | Value |
|-------|-------|
| Protocol name | Loss-Function Hypothesis Experiment (LFHE) |
| Protocol version | **v1.1** (diagnostic extensions; design unchanged) |
| Parent data protocol | CSRLE v2_boundary_safe |
| Authoritative CSRLE run | `experiments/synthetic_residual_learnability_2026-07-25_164307/` |

## Random seeds (to be frozen in config)

| Component | Seed policy |
|-----------|-------------|
| Global NumPy/TF seed | Single fixed integer (e.g. 42) — **same** for MSE and DA-MSE arms |
| Sequence shuffle | Derived from global seed |
| Bootstrap resampling | Fixed seed (e.g. 12345) for paired CIs |
| Weight initialization | TensorFlow/Keras default generator seeded |

**Rule:** MSE and DA-MSE runs differ **only** by loss function and output directory — seeds identical.

## Frozen configuration artifact (post-approval)

Path: `experiments/loss_function_hypothesis_<timestamp>/config/lfhe_config_frozen.json`

Minimum schema:

```json
{
  "protocol_version": "lfhe_v1.1",
  "lambda_disp": 1.0,
  "lambda_disp_tunable": false,
  "lambda_justification": "Frozen hypothesis-test default; see docs/loss_function_experiment/loss_function_selection.md",
  "horizon_variance_bands": ["1-12", "13-24", "25-48", "49-72", "73-96"],
  "epsilon": 1e-6,
  "csrle_experiment_dir": "experiments/synthetic_residual_learnability_2026-07-25_164307",
  "primary_condition": "b0_lc",
  "loss_control": "mse",
  "loss_treatment": "da_mse",
  "std_ddof": 0,
  "dispersion_penalty": "one_sided_log_variance",
  "global_seed": 42,
  "bootstrap_seed": 12345,
  "reproduction_tolerances": {
    "pearson_r": 0.015,
    "std_ratio": 0.015,
    "residual_mae": 0.005
  },
  "success_thresholds": {
    "mean_std_ratio_da_mse": 0.12,
    "delta_r_ci_excludes_zero": true,
    "mae_increase_max": 0.010
  }
}
```

Hyperparameters copied verbatim from Stage 2 B0 `training_metadata.json`:

- learning_rate
- batch_size
- max_epochs
- early_stopping_patience
- input_window (96)
- optimizer (Adam)

## Data immutability

| Resource | Mode |
|----------|------|
| `data/b0_lc/*.parquet` | Read-only |
| `data/ground_truth/injection_manifest.parquet` | Read-only |
| `synthetic_validation/alpha_map_b0_lc.csv` | Read-only |
| `config/synthetic_config_frozen.json` | Read-only |
| Stage 2 models / metrics | Read-only reference |

## Output directory layout (new experiment)

```
experiments/loss_function_hypothesis_<timestamp>/
├── config/
│   └── lfhe_config_frozen.json
├── b0_lc_mse/
│   ├── models/
│   ├── training/
│   └── evaluation/
├── b0_lc_da_mse/
│   ├── models/
│   ├── training/
│   └── evaluation/
├── comparisons/
│   ├── lfhe_summary.csv
│   ├── paired_mse_vs_da_mse.json
│   ├── horizon_variance_recovery_paired_bootstrap.json
│   └── energy_recovery_paired_bootstrap.json
├── plots/
│   ├── horizon_variance_recovery_mean_ci.{png,pdf}
│   ├── horizon_variance_recovery_boxplot.{png,pdf}
│   ├── horizon_variance_recovery_mse_vs_da_mse.{png,pdf}
│   ├── energy_recovery_distribution.{png,pdf}
│   └── energy_recovery_mse_vs_da_mse_scatter.{png,pdf}
├── notebooks/
│   └── (source: ../../notebooks/lfhe_visualization.ipynb)
├── verification/
│   └── mse_reproduction_gate.json
└── lfhe_final_report.json
```

## Reproduction commands (post-implementation)

```bash
# Full LFHE (after implementation approved)
python scripts/run_lfhe_experiment.py --config experiments/loss_function_hypothesis_<timestamp>/config/lfhe_config_frozen.json

# MSE control only (reproduction check)
python scripts/run_lfhe_experiment.py --config ... --arms mse_b0 --stop-after-reproduction-gate
```

*Commands are placeholders — scripts do not exist until implementation phase.*

## Metric reproducibility

All metrics computed by calling the same functions as CSRLE Stage 2:

- `utils/csrle/stage2_metrics.py` (or successor LFHE wrapper importing it)
- Paired bootstrap: same algorithm as `utils/csrle/stage2_stats.py`

Report metric function commit hash in `lfhe_final_report.json`.

## Environment

| Dependency | Notes |
|------------|-------|
| Python env | `tf_metal_env` (project standard) |
| TensorFlow/Keras | Same version as Stage 2 run |
| pandas / numpy | Same major versions as CSRLE |

Record `pip freeze` or conda export in experiment directory at run time.

## Version control

- Design docs: committed under `docs/loss_function_experiment/`.
- Experiment outputs: committed under `experiments/loss_function_hypothesis_<timestamp>/` after runs (exclude large model weights from git if project policy uses LFS — follow existing CSRLE convention).

## Audit trail

| Event | Record |
|-------|--------|
| Protocol approval date | README.md status line |
| Config freeze timestamp | `lfhe_config_frozen.json` |
| Training start/end | `training/training_metadata.json` per arm |
| Reproduction gate result | `verification/mse_reproduction_gate.json` |
| Hypothesis decision | `lfhe_final_report.json` → `hypothesis_verdict` field |
