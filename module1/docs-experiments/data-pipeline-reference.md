# Data Pipeline Reference

**Module 1:** Long-Term CPU Utilization Forecasting  
**Dataset:** Alibaba Cluster Trace (`container_usage.csv`)  
**Target metric:** `cpu_util_percent` only

This document describes how data flows through the project from raw input to final prediction.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  RAW DATA                                                       │
│  container_usage.csv                                            │
│  columns: container_id, time_stamp, cpu_util_percent, ...     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  PREPROCESSING (preprocessing.ipynb)                            │
│  Filter → Resample → Interpolate → Normalize → Feature Eng.     │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        train_df.parquet  val_df.parquet  df_resampled.parquet
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
┌──────────────────────────┐   ┌──────────────────────────────┐
│  GLOBAL GRU PATH         │   │  HYBRID PROPHET + GRU PATH   │
│  global_gru_model.ipynb  │   │  hybrid_model.ipynb          │
└──────────────────────────┘   └──────────────────────────────┘
```

---

## Stage 1: Raw Ingestion

**Notebook:** `preprocessing.ipynb`  
**Input:** `data/container_usage.csv` (1.5 GB)

| Step | Operation |
|------|-----------|
| Load CSV | `pd.read_csv('../data/container_usage.csv')` |
| Timestamp conversion | `pd.to_datetime(time_stamp, unit='s')` |
| Drop null IDs | Remove rows with null `container_id` |

---

## Stage 2: Container Filtering

**Goal:** Keep containers with sufficient, non-degenerate CPU telemetry.

| Criterion | Threshold |
|-----------|-----------|
| Minimum rows | ≥ 1100 (`total_rows`) |
| Exclude zero mean | `mean == 0` |
| Exclude zero variance | `std == 0` |
| Exclude undefined CV | `cv.isna()` |

**Result:** 486 containers retained.

**EDA (not used downstream):** Containers labeled stable / medium / spiky by coefficient of variation (CV) tertiles.

---

## Stage 3: Chronological Ordering

```python
df_final = df_final.sort_values(["container_id", "time_stamp"])
```

All subsequent operations preserve per-container time order.

---

## Stage 4: Resampling

| Parameter | Value |
|-----------|-------|
| Frequency | `15min` |
| Aggregation | Mean per container |
| Output column | `cpu_util_percent` (resampled mean) |

---

## Stage 5: Missing Value Interpolation

```python
cpu_resampled["cpu_usage"] = (
    cpu_resampled
    .groupby("container_id")["cpu_util_percent"]
    .transform(lambda x: x.interpolate())
)
```

Interpolated values stored in `cpu_usage`. Original resampled values retained in `cpu_util_percent`.

---

## Stage 6: Per-Container Normalization

**Split:** 80% train / 20% validation **per container**, chronologically.

```python
split_idx = int(len(group) * 0.8)
train_part = group.iloc[:split_idx]
val_part   = group.iloc[split_idx:]
```

**Scaler:** `MinMaxScaler(feature_range=(0, 1))`

- Fit on train portion only
- Transform both train and validation
- Output column: `cpu_scaled`

**Leakage control:** Scaler never sees validation data. ✓

---

## Stage 7: Feature Engineering

Statistics computed from **train portion only**, broadcast to both train and val:

| Feature | Description |
|---------|-------------|
| `cpu_scaled` | MinMax-normalized CPU |
| `cpu_mean` | Mean of scaled train CPU |
| `cpu_std` | Std of scaled train CPU |
| `cpu_max` | Max of scaled train CPU |
| `cpu_min` | Min of scaled train CPU |
| `hour` | Hour of day from timestamp |

**Used in models:**

| Feature | Global GRU | Hybrid GRU |
|---------|------------|------------|
| `cpu_scaled` | ✓ (target + input) | ✓ (Prophet input) |
| `cpu_mean` | ✓ | ✗ |
| `cpu_std` | ✓ | ✗ |
| `hour` | Commented out | ✗ |
| `cpu_max`, `cpu_min` | ✗ | ✗ |

---

## Stage 8: Export

| File | Rows | Containers |
|------|------|------------|
| `train_df.parquet` | 297,804 | 486 |
| `val_df.parquet` | 74,736 | 486 |
| `df_resampled.parquet` | 372,540 | 486 |

**Not exported by notebook (but required downstream):** `scalers.pkl`

---

## Stage 9: Container Selection

**Notebook:** `global_gru_model.ipynb`

```python
selected_containers = train_df["container_id"].unique()[:100]
np.save("../data/selected_containers.npy", selected_containers)
```

Both model notebooks filter to these 100 containers.

---

## Architecture 1: Hybrid Prophet + GRU

### Step A: Prophet Residual Extraction

**Notebook:** `hybrid_model.ipynb`  
**Input:** `global_train` (100 containers, train period only)

Per container:

```python
Prophet(daily_seasonality=True, weekly_seasonality=False)
residual = cpu_scaled - prophet_pred
```

Global z-score on train residuals:

```python
residual_scaled = (residual - res_mean) / res_std
```

Saved to `models/residual_stats.pkl`.

### Step B: Sequence Generation

**Function:** `create_residual_sequences()`  
**Source:** `train_residual_df` (train period only)

| Parameter | Value |
|-----------|-------|
| Input window | **288 steps** (72 hours) |
| Forecast horizon | 96 steps (24 hours) |
| Features | `residual_scaled` (1 feature) |
| Target | `residual_scaled` |

### Step C: GRU Training

| Parameter | Value |
|-----------|-------|
| Architecture | GRU(256) → GRU(128) → GRU(64) → Dense(128) → Dense(96) |
| Loss | MSE |
| Optimizer | Adam |
| Batch size | 64 |
| Epochs | 100 (EarlyStopping patience=10) |
| Shuffle | False |

**Saved:** `models/hybrid_gru.keras`

### Step D: Hybrid Prediction (Inference)

For a held-out validation period:

1. Fit Prophet on train period
2. Forecast Prophet component for validation timestamps (192 steps = 2 days)
3. **Day 1:** GRU predicts 96 residual steps from last 288 train residuals
4. **Day 2:** Slide window — drop first 96, append Day 1 predicted residuals; predict next 96
5. Combine: `final_scaled = prophet_forecast + denormalized_gru_residual`
6. Inverse MinMax transform to real CPU %

**Final formula:**

```
Final Forecast = Prophet Forecast + GRU Residual Prediction
```

---

## Architecture 2: Global GRU

### Step A: Sequence Generation

**Function:** `create_longterm_sequences()` (inline in notebook)  
**Source:** `full_df = concat(global_train, global_val)` ⚠

| Parameter | Value |
|-----------|-------|
| Input window | 96 steps (24 hours) |
| Forecast horizon | 96 steps (24 hours) |
| Features | `cpu_scaled`, `cpu_mean`, `cpu_std` (3 features) |
| Target | `cpu_scaled` |

### Step B: GRU Training

| Parameter | Value |
|-----------|-------|
| Architecture | GRU(128) → GRU(64) → Dense(64) → Dense(96) |
| Loss | MSE |
| Optimizer | Adam |
| Batch size | 256 |
| Epochs | 50 (EarlyStopping patience=3) |
| Shuffle | False |

**Not saved to disk.**

### Step C: Prediction

Direct 96-step forecast from input window. Inverse transform via per-container `MinMaxScaler` from `scalers.pkl`.

---

## Sequence Generation Logic

Shared pattern (both architectures):

```python
for cid, group in df.groupby("container_id"):
    group = group.sort_values("time_stamp")
    for i in range(len(group) - input_window - forecast_horizon):
        X = values[i : i + input_window]
        y = target_values[i + input_window : i + input_window + forecast_horizon]
```

**Implementation locations:**

| Function | Location | Default window |
|----------|----------|----------------|
| `create_residual_sequences` | `utils/sequence_utils.py` | 96 in / 96 out |
| `create_residual_sequences` | `hybrid_model.ipynb` (duplicate) | 288 in / 96 out |
| `create_longterm_sequences` | `global_gru_model.ipynb` (inline) | 96 in / 96 out |

---

## Forecast Configuration

| Parameter | Research spec | Global GRU | Hybrid GRU |
|-----------|---------------|------------|------------|
| Resampling interval | 15 min | 15 min | 15 min |
| Input window | 96 steps (24h) | 96 steps | **288 steps** |
| Forecast horizon | 96 steps (24h) | 96 steps | 96 steps |
| Containers | — | 100 | 100 |

---

## Artifact Dependency Graph

```
container_usage.csv
    └── preprocessing.ipynb
            ├── train_df.parquet ──────────────┐
            ├── val_df.parquet ────────────────┤
            └── df_resampled.parquet           │
                                                 │
selected_containers.npy ◄── global_gru_model.ipynb
scalers.pkl (manual/orphan) ◄──────────────────┤
                                                 │
                    ┌────────────────────────────┤
                    ▼                            ▼
            hybrid_model.ipynb          global_gru_model.ipynb
                    │                            │
                    ├── hybrid_gru.keras         └── (no saved model)
                    └── residual_stats.pkl
                              │
                              ▼
                      hhybrid_pred.ipynb
```

---

## Time Ranges (Preprocessed Data)

Approximate per container (100 selected):

| Split | Points | Duration |
|-------|--------|----------|
| Train | ~613 | ~153 hours (~6.4 days) |
| Val | ~154 | ~38 hours (~1.6 days) |

Total dataset span: ~8 days (1970-01-02 → 1970-01-09 in processed timestamps).

> Sufficient for daily seasonality; marginal for weekly seasonality.
