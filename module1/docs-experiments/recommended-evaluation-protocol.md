# Recommended Evaluation Protocol

**Date:** July 2026  
**Purpose:** Define a target evaluation methodology to produce thesis-quality, defensible results for Module 1  
**Status:** Recommended — not yet implemented

This protocol addresses gaps identified in the [Methodology Gap Analysis](methodology-gap-analysis.md). It should be adopted **before** reporting final Hybrid vs Global GRU comparison results.

---

## Goals

1. Eliminate validation-target leakage in Global GRU
2. Enable fair, identical evaluation between Hybrid and Global GRU
3. Support unseen container generalization claims
4. Produce reproducible, exportable experiment results
5. Align evaluation with the research specification (96-step horizon, MAE/RMSE/MAPE)

---

## Priority 1 — Fix Before Thesis Results

### 1.1 Correct Global GRU Data Split

**Problem:** Sequences built from `concat(train, val)` with 80/20 index split → ~21% of training sequences have validation-period targets.

**Required protocol:**

```
GRU training sequences  ← build from global_train ONLY
GRU early-stop val      ← last 20% of train sequences OR dedicated val sequences from train period
Final evaluation        ← global_val ONLY (preprocessing validation period)
```

**Rules:**

- Never include `global_val` rows when building training sequences
- Never split pooled train+val sequences by index for final evaluation
- Evaluate Global GRU on the same temporal holdout as Hybrid

### 1.2 Unified Evaluation Protocol

Both architectures must be evaluated with:

| Parameter | Value |
|-----------|-------|
| Containers | Same 100 (or defined holdout set) |
| Evaluation period | Preprocessing validation period (per container) |
| Forecast horizon | 96 steps (24 hours) — primary metric |
| Metrics | MAE, RMSE, MAPE in **real CPU %** |
| Aggregation | Mean ± std over all evaluation containers |

**Extended (optional):** Day 2 recursive forecast (192 steps) reported separately, not as primary.

### 1.3 Document Input Window Decision

| Option | Action |
|--------|--------|
| A. Align with spec | Change Hybrid GRU input to 96 steps |
| B. Justify extension | Keep 288 steps; document as "extended context window for residual learning" in thesis |

Either choice is valid for research — but must be **explicit and consistent** across experiments.

### 1.4 Prophet Seasonality

| Option | Action |
|--------|--------|
| A. Enable weekly | Set `weekly_seasonality=True` if data length supports it |
| B. Document exclusion | Record that ~6.4 days of train data per container is insufficient for reliable weekly seasonality |

---

## Priority 2 — Complete Research Objectives

### 2.1 Unseen Container Holdout

Required for Research Objective: *"evaluate forecasting performance on unseen containers"*

**Suggested split (486 containers):**

| Set | Containers | Purpose |
|-----|------------|---------|
| Train | 80 | GRU + Prophet training |
| Validation | 10 | Hyperparameter tuning / early stopping |
| Test | 10 | Final generalization evaluation (never seen during training) |

**Alternative:** 70 / 15 / 15 if more test containers are needed.

**Selection method:** Stratified by workload type (stable / medium / spiky) from preprocessing EDA — not `unique()[:100]`.

### 2.2 Required Metrics

Report all three preferred metrics in **real CPU %** (after inverse MinMax transform):

```
MAE  = mean(|actual - predicted|)
RMSE = sqrt(mean((actual - predicted)²))
MAPE = mean(|actual - predicted| / |actual|) × 100   (handle near-zero actuals)
```

Report per container and aggregated:

| Statistic | Description |
|-----------|-------------|
| Mean | Average across containers |
| Std | Standard deviation across containers |
| Median | Robust central tendency |
| 95% CI | Optional confidence interval |

### 2.3 Persist Both Models

| Artifact | Global GRU | Hybrid GRU |
|----------|------------|------------|
| Model weights | `models/global_gru.keras` | `models/hybrid_gru.keras` ✓ |
| Normalization stats | `models/` or `data/scalers.pkl` | `models/residual_stats.pkl` ✓ |
| Hyperparameters | JSON/YAML log | JSON/YAML log |
| Predictions | CSV/numpy per run | CSV/numpy per run |
| Metrics | JSON/CSV per run | JSON/CSV per run |

### 2.4 Experiment Directory Structure

Each experiment run should create a timestamped, non-overwriting output directory:

```
experiments/
└── 2026-07-13_hybrid-vs-global/
    ├── config.yaml              # All hyperparameters
    ├── metrics.json             # MAE, RMSE, MAPE per model
    ├── predictions/             # Per-container prediction CSVs
    ├── models/                  # Copied model artifacts
    └── plots/                   # PNG + PDF figures
        ├── actual_vs_predicted/
        ├── residuals/
        └── error_distributions/
```

---

## Priority 3 — Reproducibility

### 3.1 Random Seeds

Set at the start of every experiment:

```python
RANDOM_SEED = 42
# Python, NumPy, TensorFlow seeds
```

### 3.2 Dependency Pinning

Create `requirements.txt` from the working venv:

```
tensorflow-macos==2.16.2
prophet==1.3.0
scikit-learn==1.8.0
pandas==3.0.3
numpy==1.26.4
...
```

Regenerate `scalers.pkl` with the same sklearn version used for loading.

### 3.3 Preprocessing Artifacts

`preprocessing.ipynb` should save:

- `train_df.parquet` ✓
- `val_df.parquet` ✓
- `df_resampled.parquet` ✓
- `scalers.pkl` ✗ (add)
- `selected_containers.npy` (move here or document origin)

Remove or regenerate stale `X_*.npy` / `y_*.npy` files.

---

## Priority 4 — Visualization Standards

Per `.cursor/skills/visualization/SKILL.md`, every forecast plot should include:

- [ ] Title with model name and container ID
- [ ] Axis labels (Time, CPU Utilization %)
- [ ] Legend (Actual, Predicted, Prophet if hybrid)
- [ ] Forecast boundary line (train/val split point)
- [ ] Validation region shading
- [ ] Readable font sizes
- [ ] Saved as PNG and PDF (not notebook-only)

**Comparison plots:** Both models on identical axes, same container, same time window.

---

## Evaluation Workflow (Target State)

```
┌─────────────────────────────────────────────────────────────┐
│  1. PREPROCESSING                                           │
│     Filter → resample → normalize → export + save scalers   │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  2. CONTAINER SPLIT                                         │
│     Stratified: 80 train / 10 val / 10 test (unseen)        │
└──────────────────────────┬──────────────────────────────────┘
                           │
          ┌────────────────┴────────────────┐
          ▼                                 ▼
┌─────────────────────┐           ┌─────────────────────┐
│  3a. HYBRID TRAIN   │           │  3b. GLOBAL TRAIN   │
│  Prophet on train   │           │  Sequences from     │
│  GRU on train res.  │           │  train period only  │
└─────────┬───────────┘           └─────────┬───────────┘
          │                                 │
          └────────────────┬────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  4. EVALUATE (identical protocol)                           │
│     • Same containers                                       │
│     • Same val/test period                                  │
│     • 96-step horizon (primary)                             │
│     • MAE, RMSE, MAPE in real CPU %                         │
│     • Aggregate over all containers                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  5. REPORT                                                    │
│     • Timestamped experiment folder                           │
│     • Saved models, metrics, predictions, plots               │
│     • Strengths/weaknesses per experiment                     │
└───────────────────────────────────────────────────────────────┘
```

---

## Results Reporting Template

Use this table format for thesis/experiment reports:

### Primary Results (96-step horizon, real CPU %)

| Model | MAE | RMSE | MAPE | Containers Evaluated |
|-------|-----|------|------|---------------------|
| Hybrid Prophet + GRU | | | | 100 |
| Global GRU | | | | 100 |

### Generalization Results (unseen containers)

| Model | MAE | RMSE | MAPE | Unseen Containers |
|-------|-----|------|------|-------------------|
| Hybrid Prophet + GRU | | | | 10 |
| Global GRU | | | | 10 |

### Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | Alibaba Cluster Trace |
| Target | cpu_util_percent |
| Resampling | 15 min |
| Input window | 96 / 288 (document choice) |
| Forecast horizon | 96 steps (24 h) |
| Train/val/test containers | 80 / 10 / 10 |
| Random seed | 42 |
| Date | YYYY-MM-DD |

### Strengths and Weaknesses (required per experiment)

**Hybrid Prophet + GRU:**

- Strengths:
- Weaknesses:

**Global GRU:**

- Strengths:
- Weaknesses:

---

## Implementation Checklist

Copy and track progress:

```
Phase 1 — Methodology fixes
- [ ] Fix Global GRU: sequences from train period only
- [ ] Evaluate Global GRU on preprocessing val period
- [ ] Unify evaluation: same containers, same period, real CPU %
- [ ] Add MAPE metric
- [ ] Document 288 vs 96 input window decision
- [ ] Document Prophet weekly seasonality decision

Phase 2 — Research objectives
- [ ] Implement stratified container holdout (train/val/test)
- [ ] Evaluate on unseen test containers
- [ ] Aggregate metrics over all containers
- [ ] Save Global GRU model artifact

Phase 3 — Reproducibility
- [ ] Add random seeds
- [ ] Create requirements.txt
- [ ] Save scalers.pkl in preprocessing
- [ ] Remove/regenerate stale .npy files
- [ ] Timestamped experiment output directories
- [ ] Export plots as PNG/PDF

Phase 4 — Code quality
- [ ] Deduplicate sequence utilities
- [ ] Rename hybrid `global_model` → `hybrid_gru_model`
- [ ] Consolidate hhybrid_pred.ipynb into pipeline or document purpose
```

---

## What Not to Do

- Do not report Hybrid vs Global GRU comparison until Protocol 1.2 is satisfied
- Do not claim generalization without unseen container evaluation (2.1)
- Do not compare scaled-space MAE with real CPU % MAE
- Do not overwrite previous experiment results — use timestamped directories
- Do not shuffle sequential data during training or splitting

---

## Related Documents

- [Implementation Roadmap](implementation-roadmap.md) — prioritized action plan (fix → research → future)
- [Methodology Gap Analysis](methodology-gap-analysis.md) — detailed issue register
- [Project Audit](project-audit.md) — current repository status
- [Data Pipeline Reference](data-pipeline-reference.md) — data flow stages
