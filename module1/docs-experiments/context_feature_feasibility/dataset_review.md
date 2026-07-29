# Task 1 — Alibaba Dataset Review

**Sources inspected (read-only):** `data/train_df.parquet`, `data/val_df.parquet`, `docs/data-pipeline-reference.md`, preprocessing pipeline.

| Property | Value |
|----------|-------|
| Train rows | 296,576 |
| Validation rows | 74,428 |
| Containers (total) | 484 |
| Selected Hybrid cohort | 100 (`data/selected_containers.npy`) |
| Timestep | 15 minutes |
| Train time span | ~6.5 days per container |
| Validation time span | ~1.5 days per container |
| Timestamp origin | Alibaba trace epoch (displayed as 1970-01-* in parquet) |

---

## Column catalogue

For each column: **meaning**, **inference availability**, **current usage**, **GRU feature potential**.

### 1. `container_id` (str)

| Aspect | Detail |
|--------|--------|
| Meaning | Unique container identifier from Alibaba cluster trace |
| Inference | Known for the container being forecast |
| Currently used | Grouping key for Prophet, sequences, evaluation |
| GRU feature | **No** — categorical ID; one-hot/embedding would require new design and is not in current pipeline |

### 2. `time_stamp` (datetime64)

| Aspect | Detail |
|--------|--------|
| Meaning | 15-minute bucket timestamp |
| Inference | Known for every forecast step (future timestamps are scheduled) |
| Currently used | Chronological ordering, Prophet `ds`, hour extraction |
| GRU feature | **Indirect** — derive calendar features (hour, minute, day index) |

### 3. `cpu_util_percent` (float64, ~25.6% null)

| Aspect | Detail |
|--------|--------|
| Meaning | Raw resampled mean CPU utilization (%) before interpolation |
| Inference | **Past only** at forecast origin; future actuals unknown |
| Currently used | Source for interpolation; not direct model input |
| GRU feature | **No** — superseded by `cpu` / `cpu_scaled` |

### 4. `cpu_usage` (float64)

| Aspect | Detail |
|--------|--------|
| Meaning | Interpolated CPU utilization (%) — primary raw CPU column after gap-fill |
| Inference | Past values available |
| Currently used | Renamed/exported as `cpu` in parquet |
| GRU feature | **Low** — redundant with `cpu_scaled` after MinMax scaling |

### 5. `cpu` (float64)

| Aspect | Detail |
|--------|--------|
| Meaning | Same as `cpu_usage` (interpolated CPU %) |
| Inference | Past values available |
| Currently used | Reference raw CPU; Hybrid uses `cpu_scaled` |
| GRU feature | **Low** — use `cpu_scaled` instead for scale consistency |

### 6. `cpu_scaled` (float64)

| Aspect | Detail |
|--------|--------|
| Meaning | MinMax-scaled CPU in [0, 1]; scaler fit on train portion only |
| Inference | **Past** in-sample values available; future actuals unknown |
| Currently used | Prophet target (`y`); Global GRU input + target |
| GRU feature | **Yes (level context)** — encodes operating level/regime not present in residual alone. At each historical step: `cpu_scaled[t] = prophet_pred[t] + residual[t]` |

### 7. `cpu_mean` (float64, constant per container)

| Aspect | Detail |
|--------|--------|
| Meaning | Mean of scaled train CPU for this container (broadcast to all rows) |
| Inference | Available (train-computed constant) |
| Currently used | Global GRU input feature |
| GRU feature | **Moderate** — static container baseline; complements time-varying residual |

### 8. `cpu_std` (float64, constant per container)

| Aspect | Detail |
|--------|--------|
| Meaning | Std of scaled train CPU for this container (broadcast) |
| Inference | Available (train-computed constant) |
| Currently used | Global GRU input feature |
| GRU feature | **Yes** — container dispersion prior; helps calibrate correction magnitude (LFHE relevance) |

### 9. `cpu_max` (float64, constant per container)

| Aspect | Detail |
|--------|--------|
| Meaning | Max scaled train CPU for this container |
| Inference | Available (train constant) |
| Currently used | Not used in models |
| GRU feature | **Low** — correlated with `cpu_std`; peak threshold more actionable |

### 10. `cpu_min` (float64, constant per container)

| Aspect | Detail |
|--------|--------|
| Meaning | Min scaled train CPU (typically 0 after MinMax) |
| Inference | Available |
| Currently used | Not used |
| GRU feature | **No** — near-degenerate across cohort |

### 11. `hour` (int32, 0–23)

| Aspect | Detail |
|--------|--------|
| Meaning | Hour-of-day from `time_stamp` |
| Inference | Fully available for all past and future steps |
| Currently used | Preprocessed feature; commented out in Global GRU |
| GRU feature | **Low marginal value** — Prophet uses `daily_seasonality=True`; TMA shows lag-96 residual ACF ≈ 0.04 |

### 12. `machine_id` (str, ~98.4% null)

| Aspect | Detail |
|--------|--------|
| Meaning | Host machine identifier |
| Inference | Unknown for future deployment on new hosts |
| Currently used | Not used |
| GRU feature | **No** — sparse; not reliably available |

### 13. `mem_util_percent` (float64, ~98.4% null)

| Aspect | Detail |
|--------|--------|
| Meaning | Memory utilization % |
| Inference | Not available at 15-min resolution for most steps |
| Currently used | Not used |
| GRU feature | **No** — ~1.6% non-null overall; unusable at scale |

### 14. `cpi` (float64, ~99.8% null)

| Aspect | Detail |
|--------|--------|
| Meaning | Cycles per instruction (performance counter) |
| Inference | Not available |
| Currently used | Not used |
| GRU feature | **No** |

### 15. `mem_gps` (float64, ~99.8% null)

| Aspect | Detail |
|--------|--------|
| Meaning | Memory bandwidth (GPS) |
| Inference | Not available |
| Currently used | Not used |
| GRU feature | **No** |

### 16. `mpki` (float64, ~99.8% null)

| Aspect | Detail |
|--------|--------|
| Meaning | Misses per kilo-instruction |
| Inference | Not available |
| Currently used | Not used |
| GRU feature | **No** |

### 17. `net_in` (float64, ~98.4% null)

| Aspect | Detail |
|--------|--------|
| Meaning | Incoming network traffic |
| Inference | Not available |
| Currently used | Not used |
| GRU feature | **No** |

### 18. `net_out` (float64, ~98.4% null)

| Aspect | Detail |
|--------|--------|
| Meaning | Outgoing network traffic |
| Inference | Not available |
| Currently used | Not used |
| GRU feature | **No** |

### 19. `disk_io_percent` (float64, ~98.4% null)

| Aspect | Detail |
|--------|--------|
| Meaning | Disk I/O utilization |
| Inference | Not available |
| Currently used | Not used |
| GRU feature | **No** |

---

## Derived at runtime (not in parquet)

These are produced by the Hybrid pipeline and are relevant for context features:

| Field | Source | Inference | Currently in GRU |
|-------|--------|-----------|------------------|
| `prophet_pred` / `yhat` | Prophet fit on train, forecast on val | In-sample past + future forecast available | No |
| `residual` | `cpu_scaled − prophet_pred` | Past in-sample residuals; future unknown (target) | Yes (as `residual_scaled`) |
| `residual_scaled` | Global z-score of train residuals | Same | **Yes — sole GRU input** |
| Peak threshold (P90) | Train `cpu_scaled` quantile per container | Available from train | Used in peak-aware evaluation, not GRU input |

---

## Summary: usable information for Hybrid v2

| Category | Available? |
|----------|------------|
| Residual history (96 steps) | Yes — current input |
| Prophet level / CPU level (historical window) | Yes — derivable without leakage |
| Container train statistics (`cpu_mean`, `cpu_std`) | Yes — train constants |
| Calendar / time features | Yes — fully observable |
| Peak / regime indicators | Yes — from train thresholds + past CPU |
| Local residual volatility | Yes — causal rolling stats on past residuals |
| Multivariate telemetry (mem, net, disk, CPI) | **Effectively no** (~98%+ missing) |

The dataset **contains sufficient CPU-side information** for a modest context-enriched Hybrid. It does **not** contain usable multivariate context at 15-minute resolution.
