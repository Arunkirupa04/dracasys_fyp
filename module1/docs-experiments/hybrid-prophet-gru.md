# Hybrid Prophet + GRU — Technical Design Document

**Module:** DracaSys Module 1 — Long-Term CPU Forecasting  
**Status:** Final implementation (refactored notebook + utility modules)  
**Last updated:** July 2026

This document describes the **final** Hybrid Prophet + GRU implementation as implemented in `notebooks/hybrid_model.ipynb` and the `utils/hybrid_*.py` modules. It supersedes the original monolithic notebook workflow.

---

## 1. Module Overview

### 1.1 Purpose

The Hybrid Prophet + GRU module forecasts **long-term CPU utilization** (`cpu_util_percent`) for containerized workloads in the Alibaba Cluster Trace dataset. It supports proactive resource planning in containerized cloud environments by predicting future CPU demand before utilization increases.

The module is one of two forecasting architectures investigated in this research (the other is Global GRU). It decomposes forecasting into:

1. **Prophet** — interpretable trend and daily seasonality
2. **GRU** — nonlinear residual correction

Final hybrid forecast:

```
Hybrid Forecast = Prophet Forecast + GRU Residual Prediction
```

All predictions are ultimately reported in **real CPU percent** after inverse MinMax transformation.

### 1.2 Research Objective

Module 1 aims to:

- Develop accurate long-term CPU utilization forecasting for containerized environments
- Compare Hybrid Prophet + GRU against Global GRU under a fair evaluation protocol
- Evaluate generalization across a fixed set of selected containers
- Produce reproducible, defensible metrics suitable for thesis reporting

The Hybrid architecture specifically addresses **Research Gap 2** from the project literature review: combining interpretable statistical forecasting with nonlinear deep learning rather than relying on either approach alone.

### 1.3 Why Hybrid Prophet + GRU Was Selected

| Rationale | Explanation |
|-----------|-------------|
| Interpretability | Prophet exposes trend and daily seasonality explicitly |
| Nonlinear residuals | GRU captures patterns Prophet cannot model |
| Per-container Prophet | Each container has distinct workload shape; Prophet is fit per container at inference |
| Shared GRU | One GRU learns residual dynamics across all containers, improving scalability |
| Research alignment | Matches the documented Module 1 Architecture 1 design |

Prophet handles slow-moving structure; the GRU handles short-horizon nonlinear deviation. This division is methodologically cleaner than asking a single model to learn both components.

---

## 2. Overall Pipeline

### 2.1 End-to-End Workflow

The implementation uses a **two-section notebook workflow** controlled by `RUN_TRAINING`:

```
┌─────────────────────────────────────────────────────────────────┐
│  FROZEN PREPROCESSING (notebooks/preprocessing.ipynb)           │
│  train_df.parquet, val_df.parquet, scalers.pkl,                 │
│  selected_containers.npy                                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
   Section 1 — Training            Section 2 — Evaluation
   (RUN_TRAINING = True)            (always runs)
              │                             │
              ├─ Prophet residuals          ├─ load_hybrid_artifacts()
              ├─ GRU training               ├─ evaluate_selected_containers()
              └─ save_hybrid_artifacts()  ├─ evaluation_df / evaluation_summary
                                            └─ demo + supplementary Day 2 plots
```

**Section 1** trains the GRU and saves artifacts.  
**Section 2** loads saved artifacts and runs inference only — the GRU is **never retrained** during evaluation.

### 2.2 Inputs

| Input | Source | Role |
|-------|--------|------|
| `data/train_df.parquet` | Frozen preprocessing | Chronological train-period CPU data (scaled) |
| `data/val_df.parquet` | Frozen preprocessing | Chronological validation-period CPU data (scaled) |
| `data/scalers.pkl` | Frozen preprocessing | Per-container MinMaxScaler for inverse transform |
| `data/selected_containers.npy` | Frozen preprocessing | 100-container evaluation subset |
| `models/hybrid_gru.keras` | Section 1 output | Trained GRU weights |
| `models/residual_stats.pkl` | Section 1 output | Global residual mean, std, input window |

Training additionally uses in-memory Prophet fits per container; Prophet models are **not** persisted — they are refit at inference time from train-period data.

### 2.3 Outputs

| Output | Description |
|--------|-------------|
| `models/hybrid_gru.keras` | Saved Keras GRU model |
| `models/residual_stats.pkl` | `res_mean`, `res_std`, `input_window` |
| `evaluation_df` | Per-container Day 1 MAE, RMSE, MAPE (primary) |
| `evaluation_summary` | Aggregate mean/std/min/max over `evaluation_df` |
| `inference_results` | Dict of `HybridInferenceResult` per container (includes Day 2 arrays) |
| Demo plots | Sample container visualizations (`c_11461`) |
| Ablation outputs | Timestamped results under `experiments/` (H1.2) |

### 2.4 Data Flow Summary

```
Preprocessing outputs (scaled CPU)
        │
        ▼
Per-container Prophet fit (train period)
        │
        ▼
Residuals = cpu_scaled − prophet_pred
        │
        ▼
Global residual normalization (res_mean, res_std)
        │
        ▼
Sliding-window sequences → GRU training (Section 1 only)
        │
        ▼
Saved GRU + residual stats
        │
        ▼
Inference: Prophet forecast (val period) + GRU residual prediction
        │
        ▼
Hybrid forecast (scaled) → inverse MinMax → real CPU %
        │
        ▼
Day 1 metrics: MAE, RMSE, MAPE (multi-container aggregate)
```

### 2.5 Saved Artifacts

See [Section 7](#7-saved-artifacts) for a full artifact reference.

---

## 3. Training Pipeline

Training is implemented in `notebooks/hybrid_model.ipynb` (Section 1) and encapsulated in `utils/hybrid_training.py` for scripted use.

### 3.1 Loading Preprocessing Artifacts

```python
train_df = pd.read_parquet("../data/train_df.parquet")
val_df = pd.read_parquet("../data/val_df.parquet")
scalers = pickle.load(open("../data/scalers.pkl", "rb"))
selected_containers = np.load("../data/selected_containers.npy", allow_pickle=True)

global_train = train_df[train_df["container_id"].isin(selected_containers)]
global_val = val_df[val_df["container_id"].isin(selected_containers)]
```

Preprocessing is **frozen** — the Hybrid module reads artifacts only; it does not modify preprocessing logic or outputs.

### 3.2 Prophet Residual Generation

**Function:** `generate_prophet_residuals(df)` in `utils/hybrid_training.py`

For each container in the training dataframe:

1. Sort chronologically by `time_stamp`
2. Fit Prophet on `cpu_scaled` with:
   - `daily_seasonality=True`
   - `weekly_seasonality=False`
3. Predict in-sample (`yhat`)
4. Compute `residual = cpu_scaled − prophet_pred`

This produces `train_residual_df` with Prophet predictions and residuals appended per row.

### 3.3 Residual Normalization

Global statistics are computed across **all containers and all train timesteps**:

```python
res_mean = train_residual_df["residual"].mean()
res_std  = train_residual_df["residual"].std()
train_residual_df["residual_scaled"] = (residual - res_mean) / res_std
```

These statistics are saved in `residual_stats.pkl` and reused at inference to scale and unscale GRU residual outputs consistently.

### 3.4 Sequence Generation

**Function:** `create_residual_sequences()` in `utils/sequence_utils.py`

| Parameter | Default / current | Meaning |
|-----------|-------------------|---------|
| `input_window` | `INPUT_WINDOW` (96) | Past residual steps fed to GRU |
| `forecast_horizon` | `DAY1_HORIZON` (96) | Future residual steps predicted |
| `features` | `["residual_scaled"]` | Single-feature input |
| `target` | `"residual_scaled"` | GRU predicts normalized residuals |

For each container, all valid sliding windows are generated chronologically. Windows are pooled across containers into `X_all` (shape: `[n_sequences, input_window, 1]`) and `y_all` (shape: `[n_sequences, 96]`).

### 3.5 GRU Architecture

**Function:** `build_hybrid_gru_model(input_window, n_features, forecast_horizon)`

```
Input (input_window, 1)
    │
    ▼
GRU(256, return_sequences=True)
    │
    ▼
Dropout(0.2)
    │
    ▼
GRU(128, return_sequences=True)
    │
    ▼
Dropout(0.2)
    │
    ▼
GRU(64)
    │
    ▼
Dense(128, relu)
    │
    ▼
Dense(96)          ← DAY1_HORIZON outputs
```

The first layer `input_shape` adapts to `INPUT_WINDOW`. With the current default of 96, the model expects `(96, 1)` input tensors.

### 3.6 Training Process

| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| Loss | MSE |
| Training metric | MAE |
| Epochs | 100 (with early stopping) |
| Batch size | 64 |
| Shuffle | `False` (preserves temporal order) |
| Early stopping | `monitor="val_loss"`, `patience=10`, `restore_best_weights=True` |
| Validation split | 80/20 chronological split on **pooled sequences** |

The 80/20 sequence split is used **only for early stopping** during GRU training. Final model evaluation uses the separate preprocessing validation period (temporal holdout), not this sequence-validation split.

### 3.7 Artifact Saving

**Function:** `save_hybrid_artifacts()` in `utils/hybrid_artifacts.py`

Saves:

- `models/hybrid_gru.keras` — full Keras model
- `models/residual_stats.pkl` — dictionary:
  ```python
  {"res_mean": float, "res_std": float, "input_window": int}
  ```

---

## 4. Evaluation Pipeline

Evaluation is implemented in `notebooks/hybrid_model.ipynb` (Section 2) using `utils/hybrid_artifacts.py`, `utils/hybrid_inference.py`, and `utils/hybrid_evaluation.py`.

### 4.1 Loading Saved Artifacts

```python
hybrid_gru_model, res_mean, res_std, artifact_input_window = load_hybrid_artifacts(
    model_path=MODEL_PATH,
    residual_stats_path=RESIDUAL_STATS_PATH,
)
INPUT_WINDOW = artifact_input_window  # artifact value takes precedence
```

If the notebook `INPUT_WINDOW` differs from the saved artifact, a warning is printed and the artifact value is used. This prevents shape mismatches between saved model weights and inference tensors.

See [Section 8](#8-model-loading-workflow-production-vs-ablation) for the full production vs ablation loading workflow, including how `DEFAULT_INPUT_WINDOW`, `MODEL_PATH`, and `input_window` in `residual_stats.pkl` interact.

### 4.2 Hybrid Inference (Single Container)

**Function:** `run_hybrid_inference()` in `utils/hybrid_inference.py`  
**Return type:** `HybridInferenceResult` dataclass

Per-container inference steps:

1. **Extract** train and validation slices for `container_id`
2. **Fit Prophet** on train-period `cpu_scaled` (same config as training)
3. **Forecast** validation timestamps (up to `FORECAST_HORIZON = 192` steps)
4. **Compute train residuals** and normalize with saved `res_mean` / `res_std`
5. **Build GRU input** from last `INPUT_WINDOW` normalized train residuals
6. **Predict Day 1 residuals** (96 steps) with GRU
7. **Combine:** `day1_final_scaled = day1_prophet + day1_residual`
8. **Day 2 recursive extension** (supplementary):
   - Slide window: drop first 96 residuals, append **predicted** Day 1 residuals
   - Predict next 96 GRU residuals
   - Combine with Prophet Day 2 forecast
9. **Inverse transform** all scaled arrays to real CPU % using per-container scaler

### 4.3 Multi-Container Evaluation

**Function:** `evaluate_selected_containers()` in `utils/hybrid_evaluation.py`

Iterates all containers in `selected_containers.npy`:

1. Pre-filters evaluable containers via `_evaluable_container_ids()`
2. Runs `run_hybrid_inference()` per container
3. Computes Day 1 metrics
4. Returns `evaluation_df`, `inference_results`, `failures`, `skipped`

**Current evaluable count:** 99 of 100 selected containers.  
**Skipped:** `c_14674` — removed during frozen preprocessing but still listed in `selected_containers.npy`.

### 4.4 Primary Evaluation (Day 1)

Day 1 is the **primary research horizon**: the first 96 validation steps (24 hours at 15-minute intervals).

Metrics are computed in **real CPU percent** after inverse MinMax transform:

| Metric | Function | Formula |
|--------|----------|---------|
| MAE | `compute_day1_metrics()` | `mean(|actual − predicted|)` |
| RMSE | `compute_day1_metrics()` | `sqrt(mean((actual − predicted)²))` |
| MAPE | `compute_day1_mape()` | `mean(|actual − predicted| / max(|actual|, ε)) × 100` |

MAPE uses `MAPE_EPSILON = 0.01` percentage points on the denominator to avoid division-by-near-zero when actual CPU utilization is very small after inverse transform.

**Aggregate reporting:**

```python
evaluation_summary = summarize_evaluation_metrics(evaluation_df)
# Returns mean, std, min, max for day1_mae, day1_rmse, day1_mape
```

`evaluation_df` and `evaluation_summary` contain **Day 1 only**. Day 2 is intentionally excluded from aggregate tables.

### 4.5 Supplementary Evaluation (Day 2)

Day 2 is a **recursive 96-step extension** beyond the primary horizon:

- GRU input window slides forward using **predicted** (not actual) Day 1 residuals
- Error compounds; this is illustrative, not a primary research metric
- Computed inside `run_hybrid_inference()` and available in `HybridInferenceResult`
- Demo metrics and plots for sample container `c_11461` only
- **Not** stored in `evaluation_df` or `evaluation_summary`

---

## 5. Architecture Explanation

### 5.1 `utils/hybrid_config.py`

Central configuration constants:

| Constant | Value | Purpose |
|----------|-------|---------|
| `DEFAULT_INPUT_WINDOW` | `96` | GRU residual context length (locked after H1.2 ablation) |
| `DAY1_HORIZON` | `96` | Primary forecast horizon (24 h) |
| `FORECAST_HORIZON` | `192` | Total Prophet forecast length (Day 1 + Day 2) |
| `ABLATION_INPUT_WINDOWS` | `(96, 288)` | Candidate windows for H1.2 study |

### 5.2 `utils/hybrid_training.py`

| Function | Purpose |
|----------|---------|
| `generate_prophet_residuals(df)` | Per-container Prophet fit on train data; returns residuals |
| `build_hybrid_gru_model(...)` | Constructs Keras Sequential GRU with dynamic `input_shape` |
| `train_hybrid_gru(global_train, input_window, ...)` | End-to-end training: residuals → sequences → fit → return model + stats |

`train_hybrid_gru()` is used by `scripts/run_hybrid_input_window_ablation.py`. The notebook implements the same logic inline with `RUN_TRAINING` guards.

### 5.3 `utils/hybrid_inference.py`

| Symbol | Purpose |
|--------|---------|
| `HybridInferenceResult` | Dataclass holding all intermediate and final arrays for one container |
| `run_hybrid_inference(...)` | Full single-container pipeline: Prophet → GRU → combine → inverse transform |

Key design points:

- Prophet is refit per container at inference (not loaded from disk)
- `input_window`, `day1_horizon`, `forecast_horizon` are parameterized
- Day 2 logic uses `residual_input[day1_horizon:]` concatenated with `day1_residual_scaled`
- Returns both scaled and real-unit predictions for plotting and metric computation

### 5.4 `utils/hybrid_evaluation.py`

| Function | Purpose |
|----------|---------|
| `compute_day1_metrics()` | MAE and RMSE in real CPU % |
| `compute_day1_mape()` | Epsilon-stabilized MAPE in real CPU % |
| `_evaluable_container_ids()` | Pre-filters containers missing from data or scalers |
| `evaluate_selected_containers()` | Multi-container Day 1 evaluation loop |
| `summarize_evaluation_metrics()` | Aggregate statistics over `evaluation_df` |

### 5.5 `utils/hybrid_artifacts.py`

| Function | Purpose |
|----------|---------|
| `save_hybrid_artifacts()` | Persist Keras model + residual stats pickle |
| `load_hybrid_artifacts()` | Load model, `res_mean`, `res_std`, `input_window` |

Default paths:

- `models/hybrid_gru.keras`
- `models/residual_stats.pkl`

Older `residual_stats.pkl` files without `input_window` default to `DEFAULT_INPUT_WINDOW` on load.

### 5.6 Supporting Utility

**`utils/sequence_utils.py`** — `create_residual_sequences()`

Shared sliding-window builder used by Hybrid training. Also used by Global GRU with different feature sets.

---

## 6. Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     FROZEN PREPROCESSING OUTPUTS                         │
│  train_df.parquet   val_df.parquet   scalers.pkl   selected_containers   │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  Filter to selected    │
                    │  100 containers        │
                    │  global_train/val      │
                    └────────────┬───────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │ TRAINING (Section 1)    │ EVALUATION (Section 2)│
         ▼                       │                       ▼
┌─────────────────┐              │            ┌─────────────────────┐
│ Prophet fit per │              │            │ load_hybrid_artifacts│
│ container       │              │            │ hybrid_gru.keras     │
│ (train period)  │              │            │ residual_stats.pkl   │
└────────┬────────┘              │            └──────────┬──────────┘
         │                       │                       │
         ▼                       │                       ▼
┌─────────────────┐              │            ┌─────────────────────┐
│ residual =      │              │            │ For each container: │
│ cpu − prophet   │              │            │ Prophet fit (train) │
└────────┬────────┘              │            │ Prophet forecast    │
         │                       │            │ (val period)        │
         ▼                       │            └──────────┬──────────┘
┌─────────────────┐              │                       │
│ Global normalize│              │                       ▼
│ res_mean/std    │──────────────┼──────────► ┌─────────────────────┐
└────────┬────────┘              │            │ Last INPUT_WINDOW   │
         │                       │            │ train residuals →   │
         ▼                       │            │ GRU → Day 1 residual│
┌─────────────────┐              │            └──────────┬──────────┘
│ Sliding windows │              │                       │
│ X: (W, 1)       │              │                       ▼
│ y: (96,)        │              │            ┌─────────────────────┐
└────────┬────────┘              │            │ day1_final_scaled = │
         │                       │            │ prophet + gru_resid │
         ▼                       │            └──────────┬──────────┘
┌─────────────────┐              │                       │
│ GRU train       │              │                       ▼
│ Early stopping  │              │            ┌─────────────────────┐
└────────┬────────┘              │            │ Inverse MinMax      │
         │                       │            │ → real CPU %        │
         ▼                       │            └──────────┬──────────┘
┌─────────────────┐              │                       │
│ Save artifacts  │──────────────┘                       ▼
│ .keras + .pkl   │                          ┌─────────────────────┐
└─────────────────┘                          │ Day 1: MAE/RMSE/MAPE│
                                             │ evaluation_df       │
                                             │ evaluation_summary  │
                                             └──────────┬──────────┘
                                                        │
                                                        ▼ (supplementary)
                                             ┌─────────────────────┐
                                             │ Day 2 recursive     │
                                             │ demo container only │
                                             └─────────────────────┘

W = INPUT_WINDOW (96)
```

---

## 7. Saved Artifacts

### 7.1 Hybrid Training Outputs

| File | Format | Contents |
|------|--------|----------|
| `models/hybrid_gru.keras` | Keras SavedModel | Trained 3-layer GRU weights and architecture |
| `models/residual_stats.pkl` | Pickle dict | `res_mean`, `res_std`, `input_window` |

### 7.2 Frozen Preprocessing Outputs (Inputs)

| File | Format | Contents |
|------|--------|----------|
| `data/train_df.parquet` | Parquet | Train-period rows: `container_id`, `time_stamp`, `cpu_scaled`, features |
| `data/val_df.parquet` | Parquet | Validation-period rows (temporal holdout) |
| `data/scalers.pkl` | Pickle dict | `{container_id: MinMaxScaler}` — per-container, fit on train only |
| `data/selected_containers.npy` | NumPy array | 100 container IDs for Hybrid evaluation subset |

### 7.3 Ablation Experiment Outputs (H1.2)

Timestamped under `experiments/hybrid_input_window_ablation_<timestamp>/`:

| File | Purpose |
|------|---------|
| `w96/hybrid_gru.keras` | Model trained with 96-step input |
| `w288/hybrid_gru.keras` | Model trained with 288-step input |
| `w*/residual_stats.pkl` | Residual stats per window |
| `w*/evaluation_df.csv` | Per-container Day 1 metrics |
| `w*/evaluation_summary.csv` | Aggregate metrics |
| `comparison_table.csv` | Side-by-side 96 vs 288 comparison |
| `ablation_metadata.json` | Full experiment metadata and recommendation |

### 7.4 In-Memory Evaluation Outputs (Notebook)

| Variable | Type | Description |
|----------|------|-------------|
| `evaluation_df` | `pd.DataFrame` | Per-container Day 1 metrics |
| `evaluation_summary` | `pd.DataFrame` | Mean/std/min/max aggregates |
| `inference_results` | `dict[str, HybridInferenceResult]` | Full inference outputs per container |
| `evaluation_failures` | `list` | Containers that raised exceptions |
| `evaluation_skipped` | `list` | Containers excluded before inference |

---

## 8. Model Loading Workflow (Production vs Ablation)

This section documents the **exact** model-loading behavior implemented in `utils/hybrid_config.py`, `utils/hybrid_artifacts.py`, `utils/hybrid_training.py`, `utils/hybrid_inference.py`, `scripts/run_hybrid_input_window_ablation.py`, and `notebooks/hybrid_model.ipynb`. It clarifies how production artifacts under `models/` relate to ablation experiment artifacts under `experiments/`.

### 8.1 Input-Window Ablation Workflow

**Script:** `scripts/run_hybrid_input_window_ablation.py`

The ablation loops over `ABLATION_INPUT_WINDOWS = (96, 288)` from `utils/hybrid_config.py`:

```python
for input_window in ABLATION_INPUT_WINDOWS:
    model, res_mean, res_std, _ = train_hybrid_gru(
        global_train=global_train,
        input_window=input_window,
        verbose=1,
    )
```

For each window, `train_hybrid_gru()` in `utils/hybrid_training.py`:

1. Calls `generate_prophet_residuals(global_train)`
2. Computes global `res_mean` / `res_std`
3. Builds sequences with `create_residual_sequences(..., input_window=input_window)`
4. Builds a GRU via `build_hybrid_gru_model(input_window=...)` — first layer shape is `(input_window, 1)`
5. Trains with early stopping

Each window produces a **separate, independently trained** model.

**Save locations** (timestamped experiment directory — **not** `models/`):

```
experiments/hybrid_input_window_ablation_<timestamp>/
├── w96/
│   ├── hybrid_gru.keras
│   └── residual_stats.pkl
├── w288/
│   ├── hybrid_gru.keras
│   └── residual_stats.pkl
├── comparison_table.csv
└── ablation_metadata.json
```

Paths are built explicitly per iteration:

```python
run_dir = experiment_dir / f"w{input_window}"
model_path = run_dir / "hybrid_gru.keras"
stats_path = run_dir / "residual_stats.pkl"
save_hybrid_artifacts(
    ...,
    model_path=model_path,
    residual_stats_path=stats_path,
    input_window=input_window,
)
```

The ablation script **never writes** to `models/hybrid_gru.keras` or `models/residual_stats.pkl`.

**How evaluation selects the model:** by **explicit pairing of in-memory model + `input_window` parameter** — not by auto-routing from `input_window` alone.

Inside the ablation loop, evaluation uses the model **just trained in that iteration** (not reloaded from disk):

```python
evaluation_df, _, failures, skipped = evaluate_selected_containers(
    ...,
    hybrid_gru_model=model,       # in-memory model from this loop iteration
    res_mean=res_mean,
    res_std=res_std,
    input_window=input_window,    # passed explicitly: 96 or 288
)
```

There is no `load_hybrid_artifacts()` call during ablation evaluation. The 96-window run evaluates the 96-trained in-memory model with `input_window=96`; the 288-window run evaluates the 288-trained in-memory model with `input_window=288`.

`scripts/verify_hybrid_input_window_ablation.py` later reloads from disk using **explicit directory paths** (`w96/` vs `w288/`), not by reading `input_window` and auto-selecting a file.

#### Ablation workflow diagram

```
Frozen preprocessing (data/train_df.parquet, val_df.parquet, scalers.pkl, selected_containers.npy)
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  FOR input_window IN (96, 288):                           │
│                                                           │
│  train_hybrid_gru(global_train, input_window=...)         │
│        │                                                  │
│        ▼                                                  │
│  save to experiments/.../w{input_window}/               │
│    hybrid_gru.keras                                       │
│    residual_stats.pkl  (input_window stored inside)       │
│        │                                                  │
│        ▼                                                  │
│  evaluate_selected_containers(                            │
│    hybrid_gru_model = in-memory model  ◄── NOT reloaded │
│    input_window = input_window         ◄── explicit param │
│  )                                                        │
│        │                                                  │
│        ▼                                                  │
│  save w{input_window}/evaluation_df.csv                   │
│       w{input_window}/evaluation_summary.csv              │
└───────────────────────────────────────────────────────────┘
        │
        ▼
  comparison_table.csv
  ablation_metadata.json (recommended_input_window: 96)

  models/hybrid_gru.keras  ◄── NEVER touched by ablation script
```

### 8.2 Production Notebook — `RUN_TRAINING = True`

**Notebook:** `notebooks/hybrid_model.ipynb` configuration cell

```python
INPUT_WINDOW = DEFAULT_INPUT_WINDOW   # currently 96 in hybrid_config.py
RUN_TRAINING = True
MODEL_PATH = "../models/hybrid_gru.keras"
RESIDUAL_STATS_PATH = "../models/residual_stats.pkl"
```

When `RUN_TRAINING = True`, Section 1 trains **one** model using the notebook variable `INPUT_WINDOW` (initialized from `DEFAULT_INPUT_WINDOW`):

- `build_hybrid_gru_model(input_window=INPUT_WINDOW, ...)`
- `create_residual_sequences(..., input_window=INPUT_WINDOW, ...)`

Artifacts are saved via:

```python
save_hybrid_artifacts(
    hybrid_gru_model=global_model,
    res_mean=res_mean,
    res_std=res_std,
    model_path=MODEL_PATH,
    residual_stats_path=RESIDUAL_STATS_PATH,
    input_window=INPUT_WINDOW,
)
```

**Save location:** `models/hybrid_gru.keras` and `models/residual_stats.pkl`.

**Overwrite behavior:** `save_hybrid_artifacts()` calls `hybrid_gru_model.save(str(model_path))` with no versioning. Each training run **overwrites** the production files at those fixed paths.

### 8.3 Production Notebook — `RUN_TRAINING = False`

When `RUN_TRAINING = False`, Section 1 is skipped. Section 2 always loads from the **hardcoded production paths**:

```python
hybrid_gru_model, res_mean, res_std, artifact_input_window = load_hybrid_artifacts(
    model_path=MODEL_PATH,              # always "../models/hybrid_gru.keras"
    residual_stats_path=RESIDUAL_STATS_PATH,
)
```

`load_hybrid_artifacts()` in `utils/hybrid_artifacts.py`:

1. Loads the Keras model from `model_path` (no window-based routing)
2. Loads the pickle from `residual_stats_path`
3. Reads `input_window` from the pickle (defaults to `DEFAULT_INPUT_WINDOW` if the key is missing)
4. Returns `(model, res_mean, res_std, input_window)`

After loading, the notebook may **override** `INPUT_WINDOW` to match the pickle metadata:

```python
if artifact_input_window != INPUT_WINDOW:
    print("Warning: notebook INPUT_WINDOW=... does not match artifact input_window=...")
    INPUT_WINDOW = artifact_input_window
```

**Key points:**

| Question | Answer |
|----------|--------|
| Which model is loaded? | Always `models/hybrid_gru.keras` (unless `MODEL_PATH` is manually changed) |
| Does changing `DEFAULT_INPUT_WINDOW` load a different file? | **No** — it only sets the initial `INPUT_WINDOW` variable before load |
| Does the notebook auto-load experiment models (`experiments/.../w96/`)? | **No** — experiment paths are never referenced by the notebook |

#### Production workflow diagram

```
Frozen preprocessing
  data/train_df.parquet, val_df.parquet, scalers.pkl, selected_containers.npy
        │
        ├──────────────────────────────────────┐
        │ RUN_TRAINING = True                  │ RUN_TRAINING = False
        ▼                                      │ (Section 1 skipped)
  generate_prophet_residuals()                 │
  normalize residuals (res_mean, res_std)      │
  create_residual_sequences(INPUT_WINDOW)      │
  build_hybrid_gru_model(INPUT_WINDOW)         │
  train GRU                                    │
        │                                      │
        ▼                                      │
  save_hybrid_artifacts(                       │
    model_path = ../models/hybrid_gru.keras  ◄── OVERWRITES production
    residual_stats_path = ../models/residual_stats.pkl
    input_window = INPUT_WINDOW
  )                                            │
        │                                      │
        └────────────────┬─────────────────────┘
                         ▼
              load_hybrid_artifacts(
                model_path = ../models/hybrid_gru.keras   ◄── ALWAYS this path
                residual_stats_path = ../models/residual_stats.pkl
              )
                         │
                         ▼
              artifact_input_window from pickle
              → overrides INPUT_WINDOW if mismatch
                         │
                         ▼
              evaluate_selected_containers(
                hybrid_gru_model = loaded model
                input_window = INPUT_WINDOW (from artifact)
              )
                         │
                         ▼
              evaluation_df / evaluation_summary
```

### 8.4 Purpose of `input_window` in `residual_stats.pkl`

**Saved** by `save_hybrid_artifacts()`:

```python
pickle.dump({
    "res_mean": res_mean,
    "res_std": res_std,
    "input_window": input_window,
}, handle)
```

**Loaded** by `load_hybrid_artifacts()`:

```python
input_window = int(residual_stats.get("input_window", DEFAULT_INPUT_WINDOW))
return (hybrid_gru_model, res_mean, res_std, input_window)
```

| Role | Used? | Details |
|------|-------|---------|
| Metadata only | Partially | Records which window was used during training |
| Select which `.keras` file to load | **No** | Model selection is entirely determined by the `model_path` argument |
| Configure inference tensor shape | **Yes** | Passed to `evaluate_selected_containers()` → `run_hybrid_inference()` |

At inference, `input_window` drives tensor construction in `utils/hybrid_inference.py`:

```python
residual_input = train_container["residual_scaled"].values[-input_window:]
X_input = X_input.reshape(1, input_window, 1)
# Day 2:
day2_input_residuals = np.concatenate([residual_input[day1_horizon:], day1_residual_scaled])
X_day2 = day2_input_residuals.reshape(1, input_window, 1)
```

The Keras model itself also encodes input shape in its first layer. A 288-window model expects `(None, 288, 1)`; a 96-window model expects `(None, 96, 1)`. The loaded weights and the `input_window` used at inference **must agree**, or `predict()` will fail with a shape error.

### 8.5 Promoting the Ablation Winner to Production

After H1.2 selected the 96-step input window, the ablation artifacts live at:

```
experiments/hybrid_input_window_ablation_2026-07-13_155559/w96/hybrid_gru.keras
experiments/hybrid_input_window_ablation_2026-07-13_155559/w96/residual_stats.pkl
```

`DEFAULT_INPUT_WINDOW = 96` is already set in `utils/hybrid_config.py`. **Changing this constant alone does not promote the experiment model to production.**

| Step | Required? | Effect |
|------|-----------|--------|
| Set `DEFAULT_INPUT_WINDOW = 96` | Already done | Configures notebook/training default only |
| Copy or retrain into `models/` | **Yes** | Production load path points only here |
| Overwrite `models/hybrid_gru.keras` | **Yes** | With the 96-window trained weights |
| Overwrite `models/residual_stats.pkl` | **Yes** | With `res_mean`, `res_std`, `input_window=96` from the 96 run |
| Retrain via notebook (`RUN_TRAINING=True`) | Alternative to copy | Trains a fresh 96-window model directly into `models/` |

**Copy approach:** copy `w96/hybrid_gru.keras` → `models/hybrid_gru.keras` and `w96/residual_stats.pkl` → `models/residual_stats.pkl`.

**Retrain approach:** set `RUN_TRAINING = True`, ensure `INPUT_WINDOW = 96`, run Section 1 — overwrites `models/` automatically.

**Files that remain untouched:** everything under `experiments/hybrid_input_window_ablation_*/` (`w96/`, `w288/`, `comparison_table.csv`, `ablation_metadata.json`). The ablation script never writes to `models/`; experiment directories are independent archives.

### 8.6 FAQ: Does `DEFAULT_INPUT_WINDOW` Auto-Load the Experiment Model?

**Question:** If I change `DEFAULT_INPUT_WINDOW` from 288 to 96 while `RUN_TRAINING=False`, will the notebook automatically load the 96-window experiment model?

**Answer: NO**

**Why (implementation references):**

1. **Model loading is path-based, not window-based.** The notebook always calls `load_hybrid_artifacts(model_path="../models/hybrid_gru.keras", residual_stats_path="../models/residual_stats.pkl")`. `DEFAULT_INPUT_WINDOW` is never passed to `load_hybrid_artifacts()` and is never used to construct a path like `experiments/.../w96/hybrid_gru.keras`.

2. **`DEFAULT_INPUT_WINDOW` only initializes the notebook variable** before load: `INPUT_WINDOW = DEFAULT_INPUT_WINDOW`. Changing it from 288 to 96 sets `INPUT_WINDOW = 96` initially, but does not change which file is loaded.

3. **After load, the pickle can override `INPUT_WINDOW` anyway.** If `models/residual_stats.pkl` still contains `input_window: 288` (from an old production train), `INPUT_WINDOW` becomes 288 regardless of `DEFAULT_INPUT_WINDOW = 96`.

4. **The experiment 96-model lives at a different path** that the notebook never references: `experiments/hybrid_input_window_ablation_<timestamp>/w96/hybrid_gru.keras`. Changing `DEFAULT_INPUT_WINDOW` does not point `MODEL_PATH` there.

5. **Even if `input_window` in the pickle were 96**, the notebook would still load whatever weights are in `models/hybrid_gru.keras`. If that file contains a 288-window model, inference would use 96-step tensors against a 288-input model (or vice versa) — a shape mismatch at `hybrid_gru_model.predict()`, not an automatic swap to the experiment artifact.

**Bottom line:** `DEFAULT_INPUT_WINDOW` controls the training default and the initial notebook variable. Production evaluation always loads `models/hybrid_gru.keras` + `models/residual_stats.pkl`. Promoting the ablation winner requires explicitly copying or retraining into `models/`.

---

## 9. Configuration

### 9.1 Forecast Horizons

| Parameter | Location | Value | Description |
|-----------|----------|-------|-------------|
| `INPUT_WINDOW` | `hybrid_config.py` / notebook | `96` | GRU input residual steps |
| `DAY1_HORIZON` | `hybrid_config.py` | `96` | Primary forecast steps (24 h) |
| `FORECAST_HORIZON` | `hybrid_config.py` | `192` | Prophet total forecast (Day 1 + Day 2) |

Resampling interval: **15 minutes** (from frozen preprocessing).  
96 steps = 24 hours.

### 9.2 Prophet Parameters

Applied identically in training residual generation and inference:

```python
Prophet(
    daily_seasonality=True,
    weekly_seasonality=False,
)
```

No explicit `yearly_seasonality` argument (Prophet default applies).

### 9.3 GRU Hyperparameters

| Parameter | Value |
|-----------|-------|
| Layer 1 | GRU(256, return_sequences=True) |
| Layer 2 | GRU(128, return_sequences=True) |
| Layer 3 | GRU(64) |
| Dropout | 0.2 after each recurrent layer |
| Dense hidden | 128 (ReLU) |
| Output | 96 (linear, one per Day 1 step) |
| Optimizer | Adam |
| Loss | MSE |
| Epochs | 100 max |
| Batch size | 64 |
| Early stopping patience | 10 |

### 9.4 Evaluation Parameters

| Parameter | Value |
|-----------|-------|
| `MAPE_EPSILON` | `0.01` (percentage points) |
| Selected containers | 100 |
| Evaluable containers | 99 (current frozen data) |
| Demo container | `c_11461` |

### 9.5 Notebook Control Flags

| Flag | Default | Effect |
|------|---------|--------|
| `RUN_TRAINING` | `False` | `True`: run Section 1 training; `False`: skip training, evaluate saved model |
| `MODEL_PATH` | `../models/hybrid_gru.keras` | GRU model path |
| `RESIDUAL_STATS_PATH` | `../models/residual_stats.pkl` | Residual stats path |

---

## 10. Research Decisions

### 10.1 Why Preprocessing Is Frozen

- Ensures Hybrid, Global GRU, and future experiments share identical train/val splits and scalers
- Prevents accidental data leakage from re-running preprocessing with different parameters
- Makes evaluation reproducible: any model change is isolated from data pipeline changes
- Preprocessing decisions are documented separately in `docs/data-pipeline-reference.md`

### 10.2 Why Day 1 Is Primary

- Matches the research specification: 96-step (24-hour) forecast horizon
- Uses a clean temporal holdout (preprocessing validation period) without recursive error injection
- Produces a single defensible metrics table (`evaluation_df`) for thesis comparison
- Aligns with the recommended evaluation protocol in `docs/recommended-evaluation-protocol.md`

### 10.3 Why Day 2 Is Supplementary

- Requires **predicted** Day 1 residuals as GRU input — errors compound beyond 96 steps
- Exceeds the primary 96-step research horizon
- Useful for demonstrating error growth, but not suitable as the primary comparison metric
- Computed and visualized for demo container only; excluded from `evaluation_df`

### 10.4 Why Training and Evaluation Are Separated

- **No GRU retraining during evaluation** — prevents accidental leakage and ensures saved-model reproducibility
- Evaluation can run independently with `RUN_TRAINING = False`
- Mirrors production deployment: train once, evaluate many times
- Verified by `scripts/verify_hybrid_train_eval_split.py` (in-memory vs loaded model bit-exact equivalence)

### 10.5 Why the Reusable Inference Module Exists

- Eliminates duplicated inline inference code across notebook cells
- Provides a single canonical implementation verified by `scripts/verify_hybrid_inference.py`
- Returns structured `HybridInferenceResult` for both evaluation and visualization
- Enables scripted multi-container evaluation without notebook execution

### 10.6 Why Multi-Container Evaluation Was Introduced

- Single-container demo metrics (`c_11461`) are not representative of model performance
- Research requires aggregate statistics (mean ± std) over the evaluation cohort
- Enables fair comparison with Global GRU on the same 100-container subset
- Pre-filtering (`_evaluable_container_ids`) reports skipped containers transparently

### 10.7 Input Window Decision (H1.2 Ablation)

An ablation compared `INPUT_WINDOW ∈ {96, 288}` with identical preprocessing, containers, and Day 1 protocol.

| Window | Mean MAE | Mean RMSE | Mean MAPE |
|--------|----------|-----------|-----------|
| **96** | **1.7402** | **2.3792** | 112.07 |
| 288 | 1.7523 | 2.3920 | 112.20 |

**Decision:** 96-step input window selected — lower mean MAE and RMSE, aligns with the 24-hour research specification, simpler input shape.  
`DEFAULT_INPUT_WINDOW = 96` in `utils/hybrid_config.py`.

See [Section 8.5](#85-promoting-the-ablation-winner-to-production) for how to promote the ablation winner into `models/`, and [Section 8.6](#86-faq-does-default_input_window-auto-load-the-experiment-model) for why changing `DEFAULT_INPUT_WINDOW` alone does not auto-load experiment artifacts.

---

## 11. Final Folder Structure

```
FYP_Long_Term/
├── data/                                    # Frozen preprocessing outputs
│   ├── train_df.parquet
│   ├── val_df.parquet
│   ├── scalers.pkl
│   ├── selected_containers.npy
│   └── preprocessing_summary.json
│
├── models/                                  # Production Hybrid artifacts
│   ├── hybrid_gru.keras
│   └── residual_stats.pkl
│
├── experiments/                             # Timestamped experiment outputs
│   └── hybrid_input_window_ablation_*/
│       ├── w96/
│       ├── w288/
│       ├── comparison_table.csv
│       └── ablation_metadata.json
│
├── notebooks/
│   ├── preprocessing.ipynb                  # Frozen — produces data/ artifacts
│   └── hybrid_model.ipynb                   # Final Hybrid train + evaluate notebook
│
├── utils/
│   ├── hybrid_config.py                     # Forecast constants
│   ├── hybrid_training.py                   # Prophet residuals + GRU training
│   ├── hybrid_inference.py                  # Single-container inference
│   ├── hybrid_evaluation.py                 # Multi-container Day 1 evaluation
│   ├── hybrid_artifacts.py                  # Save/load model + stats
│   └── sequence_utils.py                    # Sliding-window sequence builder
│
├── scripts/
│   ├── run_hybrid_input_window_ablation.py  # H1.2: train + compare 96 vs 288
│   ├── verify_hybrid_inference.py           # Inference correctness vs legacy
│   ├── verify_hybrid_evaluation.py            # Multi-container + MAPE checks
│   ├── verify_hybrid_train_eval_split.py      # Train/load equivalence
│   ├── verify_hybrid_h1_5.py                  # Primary vs supplementary separation
│   └── verify_hybrid_input_window_ablation.py # Ablation reproducibility
│
└── docs/
    ├── hybrid-prophet-gru.md                  # This document
    ├── data-pipeline-reference.md
    ├── recommended-evaluation-protocol.md
    └── implementation-roadmap.md
```

**Note:** `notebooks/hhybrid_pred.ipynb` is a legacy inference notebook from an earlier implementation phase. The final workflow uses `hybrid_model.ipynb` Section 2 with `utils/hybrid_inference.py`.

---

## 12. Reproducibility

### 12.1 Retrain the Model

1. Ensure frozen preprocessing artifacts exist in `data/`
2. Open `notebooks/hybrid_model.ipynb`
3. Set `RUN_TRAINING = True` in the configuration cell
4. Confirm `INPUT_WINDOW = DEFAULT_INPUT_WINDOW` (96)
5. Run all cells through Section 1
6. Artifacts saved to `models/hybrid_gru.keras` and `models/residual_stats.pkl`

To promote a specific ablation run (e.g. `w96/`) into production without retraining, copy its artifacts into `models/` — see [Section 8.5](#85-promoting-the-ablation-winner-to-production).

To re-run the input-window ablation (saves to `experiments/`, **not** `models/`):

```bash
python scripts/run_hybrid_input_window_ablation.py
```

### 12.2 Evaluate a Saved Model

1. Set `RUN_TRAINING = False`
2. Run configuration and data-loading cells
3. Run Section 2 (loads artifacts, runs `evaluate_selected_containers()`)
4. Inspect `evaluation_df` and `evaluation_summary`

### 12.3 Evaluation-Only Mode (No Training)

This is the default notebook configuration:

```python
RUN_TRAINING = False
```

Section 1 cells are guarded by `if RUN_TRAINING:` and skipped. Section 2 always loads artifacts and evaluates.

### 12.4 Reproduce Reported Metrics

Run verification scripts after training or loading:

```bash
python scripts/verify_hybrid_evaluation.py
python scripts/verify_hybrid_inference.py
python scripts/verify_hybrid_train_eval_split.py
python scripts/verify_hybrid_h1_5.py
```

These scripts use the same frozen `data/` artifacts and `models/` weights. They assert:

- 99 containers evaluated, 1 skipped (`c_14674`)
- Day 1 metrics consistent with inference outputs
- Loaded model predictions identical to in-memory model
- `evaluation_df` contains Day 1 only (no Day 2 columns)

### 12.5 Reproduce Input Window Ablation

```bash
python scripts/run_hybrid_input_window_ablation.py
python scripts/verify_hybrid_input_window_ablation.py
```

Outputs are written to a new timestamped `experiments/hybrid_input_window_ablation_<timestamp>/` directory without overwriting production `models/`.

### 12.6 Stale Module Cache (Jupyter)

After editing `utils/` modules, reload before importing:

```python
import importlib
import utils.hybrid_training as hybrid_training
importlib.reload(hybrid_training)
```

The notebook configuration cell includes reload logic for this reason. If imports fail after code changes, restart the Jupyter kernel.

---

## 13. Known Limitations

| Limitation | Detail |
|------------|--------|
| Short training history | ~613 timesteps per container (~6.4 days at 15-minute intervals) |
| Weekly seasonality disabled | Insufficient history for reliable weekly patterns; `weekly_seasonality=False` |
| Selected subset evaluation | Metrics reported over 100 selected containers, not all 484 preprocessed containers |
| Stale container ID | `c_14674` in `selected_containers.npy` but removed from frozen preprocessing — always skipped |
| Prophet refit cost | Per-container Prophet fit at inference is computationally expensive for large cohorts |
| MAPE sensitivity | Near-zero actual CPU values can inflate MAPE despite epsilon stabilization; high std MAPE expected |
| No persisted Prophet models | Prophet is refit from scratch per container per inference run |
| Single global GRU | One GRU serves all containers; no per-container fine-tuning |
| Day 2 error compounding | Recursive Day 2 uses predicted residuals; not suitable for primary comparison |
| Stochastic training | GRU weights may vary slightly between runs even with fixed seeds (hardware-dependent) |

---

## 14. Future Improvements

The following are identified future work items. **None are implemented in the current codebase.**

| Area | Potential improvement |
|------|----------------------|
| Evaluation cohort | Evaluate all 484 preprocessed containers or a held-out unseen-container split |
| Prophet persistence | Cache or persist per-container Prophet models to reduce inference latency |
| Weekly seasonality | Re-enable when longer training history is available |
| Cross-architecture comparison | Unified evaluation script comparing Hybrid and Global GRU on identical protocol |
| Experiment logging | Timestamped metrics, hyperparameters, and plots for every training run |
| Random seeds | Explicit Python/NumPy/TensorFlow seeds in notebook and scripts |
| Production deployment | REST inference endpoint wrapping `run_hybrid_inference()` |
| Multi-resource forecasting | Extend beyond `cpu_util_percent` to memory, disk, etc. |
| Online adaptation | Incremental residual model updates as new telemetry arrives |
| Alternative horizons | Configurable forecast horizons beyond 96 steps |

---

## Appendix A: `HybridInferenceResult` Fields

| Field | Description |
|-------|-------------|
| `day1_final_real` | Hybrid Day 1 prediction (real CPU %) |
| `actual_day1_real` | Ground truth Day 1 (real CPU %) |
| `prophet_day1_real` | Prophet-only Day 1 (real CPU %) |
| `day2_final_real` | Hybrid Day 2 recursive prediction (real CPU %) |
| `actual_day2_real` | Ground truth Day 2 (real CPU %) |
| `day1_residual_scaled` | GRU Day 1 residual output (normalized) |
| `residual_input` | Last `INPUT_WINDOW` normalized train residuals |
| `input_window` | Input window used for this inference run |
| `n_train_steps` | Number of train-period timesteps for container |
| `n_val_steps` | Number of validation-period timesteps for container |

---

## Appendix B: `evaluation_df` Schema

| Column | Type | Description |
|--------|------|-------------|
| `container_id` | str | Container identifier |
| `day1_mae` | float | Day 1 MAE (real CPU %) |
| `day1_rmse` | float | Day 1 RMSE (real CPU %) |
| `day1_mape` | float | Day 1 MAPE (%) |
| `train_steps` | int | Train-period length |
| `validation_steps` | int | Validation-period length |

---

## Appendix C: Related Documentation

- [Data Pipeline Reference](data-pipeline-reference.md) — full preprocessing-to-prediction flow
- [Recommended Evaluation Protocol](recommended-evaluation-protocol.md) — target methodology
- [Implementation Roadmap](implementation-roadmap.md) — phased project plan
- [Project Audit](project-audit.md) — repository status snapshot

---

*This document reflects the implementation as of July 2026. Update when `DEFAULT_INPUT_WINDOW`, evaluation protocol, or artifact formats change.*
