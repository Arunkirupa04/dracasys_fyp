# Module 1 Project Audit

**Date:** July 2026  
**Scope:** Read-only audit of Module 1 (Long-Term CPU Forecasting)  
**Repository:** `FYP_Long_Term`

---

## Executive Summary

Module 1 is implemented as a **working research prototype** in Jupyter notebooks. Both target architectures exist:

1. **Hybrid Prophet + GRU** — trained, evaluated, and partially persisted
2. **Global GRU** — trained and evaluated in-notebook only

The data pipeline from Alibaba trace → resampled/normalized parquet is sound. Main gaps are **engineering maturity** (modularity, reproducibility), **artifact consistency**, **code duplication**, and **evaluation rigor**.

---

## Repository Structure

```
FYP_Long_Term/
├── .cursor/skills/          # Project documentation & agent guidance
├── data/                    # Gitignored (~2.8 GB)
├── docs/                    # This documentation
├── models/                  # Partially untracked model artifacts
├── notebooks/               # Core implementation (4 notebooks)
├── utils/                   # Single shared Python module
├── tf_metal_env/            # Gitignored local venv
└── .gitignore
```

### What Exists vs. What Is Missing

| Category | Present | Missing |
|----------|---------|---------|
| Notebooks | 4 | — |
| Python modules | 1 (`utils/sequence_utils.py`) | `src/` packages (data, preprocessing, training, evaluation, config) |
| Documentation | Cursor skills + `docs/` | Root `README.md` |
| Dependencies | Local venv only | `requirements.txt` / `pyproject.toml` |
| Tests | None | Unit/integration tests |
| Experiment outputs | Hybrid model only | Timestamped experiment dirs, Global GRU checkpoint |

---

## Git Tracking Status

**Tracked (initial commit):**

- `.gitignore`
- `notebooks/preprocessing.ipynb`
- `notebooks/global_gru_model.ipynb`
- `notebooks/hybrid_model.ipynb`
- `utils/sequence_utils.py`

**Untracked / modified:**

- `notebooks/hhybrid_pred.ipynb`
- `notebooks/hybrid_model.ipynb` (modified)
- `models/hybrid_gru.keras`, `models/residual_stats.pkl`
- `.cursor/skills/`
- All of `data/` (gitignored)

---

## Module 1 File Inventory

### Core Implementation

| File | Role |
|------|------|
| `notebooks/preprocessing.ipynb` | Raw data → cleaned, resampled, normalized parquet outputs |
| `notebooks/global_gru_model.ipynb` | Architecture 2: Global GRU training & evaluation |
| `notebooks/hybrid_model.ipynb` | Architecture 1: Prophet residuals → GRU training, 2-day forecast, model save |
| `notebooks/hhybrid_pred.ipynb` | Inference-only hybrid evaluation (loads saved model) |
| `utils/sequence_utils.py` | Reusable sliding-window sequence builder |

### Data Artifacts (`data/` — gitignored)

| File | Size (approx.) | Purpose |
|------|----------------|---------|
| `container_usage.csv` | 1.5 GB | Raw Alibaba Cluster Trace |
| `df_resampled.parquet` | 1.1 MB | Post-resample/interpolation (486 containers) |
| `train_df.parquet` | 1.9 MB | Chronological train split (297,804 rows) |
| `val_df.parquet` | 485 KB | Chronological val split (74,736 rows) |
| `selected_containers.npy` | 1.6 KB | 100 container IDs used by model notebooks |
| `scalers.pkl` | 132 KB | 486 per-container `MinMaxScaler` objects |
| `X_train.npy`, `y_train.npy` | ~1.1 GB | Pre-built sequences — **likely stale** |
| `X_val.npy`, `y_val.npy` | ~272 MB | Pre-built validation sequences — **likely stale** |

### Model Artifacts (`models/`)

| File | Purpose |
|------|---------|
| `hybrid_gru.keras` | Trained hybrid residual GRU (288→96) |
| `residual_stats.pkl` | Global residual mean/std for inverse scaling |

**Missing:** Global GRU saved model (no `.keras` / `.h5` export).

---

## End-to-End Workflow

```
container_usage.csv
        │
        ▼
preprocessing.ipynb
        │
        ├── train_df.parquet
        ├── val_df.parquet
        └── df_resampled.parquet
        │
        ├──────────────────────────┐
        ▼                          ▼
global_gru_model.ipynb      hybrid_model.ipynb
        │                          │
        │                          ├── models/hybrid_gru.keras
        │                          └── models/residual_stats.pkl
        │                                    │
        ▼                                    ▼
In-notebook predictions only      hhybrid_pred.ipynb (inference)
```

---

## Pipeline Stage Summary

| Stage | Location | Key Output |
|-------|----------|------------|
| Raw ingestion | `preprocessing.ipynb` | Parsed timestamps |
| Container filtering | `preprocessing.ipynb` | 486 containers (≥1100 rows, non-degenerate) |
| Resampling | `preprocessing.ipynb` | 15-minute intervals |
| Interpolation | `preprocessing.ipynb` | `cpu_usage` column |
| Normalization | `preprocessing.ipynb` | `cpu_scaled` (MinMax, train-fit only) |
| Feature engineering | `preprocessing.ipynb` | `cpu_mean`, `cpu_std`, `cpu_max`, `cpu_min`, `hour` |
| Container selection | `global_gru_model.ipynb` | 100 containers → `selected_containers.npy` |
| Prophet residuals | `hybrid_model.ipynb` | `residual`, `residual_scaled` |
| Sequence generation | Model notebooks | Sliding windows |
| GRU training | Model notebooks | Hybrid saved; Global not saved |
| Evaluation | Model notebooks | MAE/RMSE in notebooks |

---

## Key Data Dimensions

| Artifact | Value |
|----------|-------|
| Preprocessed containers | 486 |
| Model training containers | 100 |
| Train rows (selected 100) | 61,272 (~613/container) |
| Val rows (selected 100) | 15,373 (~154/container) |
| Global GRU sequences | 57,445 total (96×3 input, 96 output) |
| Hybrid GRU sequences | Built in-notebook (288×1 input, 96 output) |
| Stale `X_train.npy` | `(148732, 288, 3)` — does not match current hybrid (288, 1) |

---

## Reported Metrics (Notebook Outputs)

| Model | Context | MAE | RMSE |
|-------|---------|-----|------|
| Hybrid | Day 1, single container, val holdout | 1.47 | 2.03 |
| Hybrid | Day 2, recursive | 1.85 | 2.45 |
| Global GRU | Scaled validation space | ~0.016 | — |
| Global GRU | Single container, real CPU % | ~1.99 | ~2.47 |

> These metrics are **not directly comparable** — see [Methodology Gap Analysis](methodology-gap-analysis.md).

---

## Issues Identified

### Duplicated Code

- `create_residual_sequences` in `utils/sequence_utils.py` imported then **redefined inline** in `hybrid_model.ipynb`
- `create_longterm_sequences` duplicated in `global_gru_model.ipynb` instead of shared utility
- Hybrid inference block duplicated in `hybrid_model.ipynb` and `hhybrid_pred.ipynb`
- Hybrid GRU model variable named `global_model` — confusing vs. Global GRU architecture

### Unused / Dead Code

- TensorFlow/Keras imports in `preprocessing.ipynb` (never used)
- `hour`, `cpu_max`, `cpu_min` engineered but unused in models
- Commented `.npy` loaders and alternate val paths in Global GRU notebook

### Orphaned / Stale Artifacts

| Artifact | Issue |
|----------|-------|
| `scalers.pkl` | Required by model notebooks but **not written** by `preprocessing.ipynb` |
| `X_train.npy` (288, **3** features) | Does not match current hybrid (288, **1** feature) |
| `X_*.npy`, `y_*.npy` | No notebook currently writes these; likely from superseded experiment (May 15) |

### Incomplete Relative to Research Spec

| Gap | Details |
|-----|---------|
| Input window | Spec: 96 steps; Hybrid uses 288 for residual context |
| Global GRU persistence | Trained but never saved |
| Unseen container evaluation | Not implemented |
| MAPE | Listed as preferred metric; not computed |
| Reproducibility | No random seeds, hyperparameter logs, timestamped outputs |
| Modular Python | Only 1 utility file; logic lives in notebooks |
| Saved plots | Embedded in notebooks only; not exported as PNG/PDF |

---

## Alignment with Research Objectives

| Objective | Status |
|-----------|--------|
| Accurate long-term CPU forecasting | Both models produce reasonable results (~1.5–2.0 MAE) |
| Compare Hybrid vs Global GRU | Both implemented; informal notebook comparison only |
| Evaluate on unseen containers | **Not yet** |
| Robustness across workload patterns | EDA labels exist; not used in modeling |
| Modular forecasting pipeline | **Partial** — preprocessing separated; modeling monolithic |

---

## Dependencies (Local Venv)

No `requirements.txt` exists. Core stack from `tf_metal_env/`:

| Package | Version |
|---------|---------|
| Python | 3.11.15 |
| TensorFlow | 2.16.2 |
| Prophet | 1.3.0 |
| scikit-learn | 1.8.0 |
| pandas | 3.0.3 |
| numpy | 1.26.4 |

**Warning:** `scalers.pkl` was pickled with sklearn 1.6.1 but loaded with 1.8.0.

---

## Next Steps

See [Implementation Roadmap](implementation-roadmap.md) for the prioritized action plan, and [Recommended Evaluation Protocol](recommended-evaluation-protocol.md) for evaluation methodology details.
