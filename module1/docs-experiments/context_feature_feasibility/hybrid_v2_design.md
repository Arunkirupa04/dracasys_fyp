# Task 7 — Hybrid v2 Input Design

Proposed context-enriched input for a **future** Hybrid experiment. This document specifies tensor layout only — no implementation in this feasibility stage.

---

## Current Hybrid (v1)

```
Input X: float32[B, 96, 1]
Channel 0: residual_scaled[t-95 : t]   (global z-score from train residuals)

Target y: float32[B, 96]
         residual_scaled[t+1 : t+96]
```

Each sequence: last 96 scaled Prophet residuals → predict next 96 scaled residuals.

---

## Proposed Hybrid v2 (compact)

```
Input X: float32[B, 96, 5]
```

For each timestep `i` in the input window (`i = t-95, …, t`), aligned rows:

| Channel | Symbol | Description | Source at inference |
|---------|--------|-------------|---------------------|
| 0 | `r_scaled[i]` | Scaled Prophet residual | Past in-sample residuals |
| 1 | `yhat_scaled[i]` | Scaled Prophet level | Prophet in-sample fit on past |
| 2 | `vol_12[i]` | Rolling std of **past** residuals, w=12 | Causal rolling on r |
| 3 | `cpu_std_c` | Container train std (constant) | Train stat, broadcast |
| 4 | `peak[i]` | 1 if cpu_scaled[i] ≥ train P90 | Past CPU + train threshold |

### Tensor illustration (single sample)

```
Timestep:     t-95   t-94   t-93   ...   t-1    t
              ─────────────────────────────────────
ch0 residual   -0.31  -0.28  -0.25  ...  +0.12  +0.15
ch1 yhat        0.42   0.44   0.45  ...   0.71   0.73
ch2 vol_12      0.08   0.08   0.09  ...   0.11   0.12
ch3 cpu_std     0.17   0.17   0.17  ...   0.17   0.17   ← broadcast
ch4 peak        0      0      0     ...    1      1
```

Shape: **`(96, 5)`** per sequence, batch **`(B, 96, 5)`**.

---

## Scaling rules (leakage-safe)

| Feature | Fit stats on | Transform |
|---------|--------------|-----------|
| `residual_scaled` | Train residuals (global) | Existing pipeline |
| `prophet_yhat_scaled` | Train prophet_yhat per container OR global | `(yhat - μ_train) / σ_train` |
| `vol_12` | Optional: divide by train residual std | `vol_12 / σ_res_train` |
| `cpu_std` | Already train-computed | Use as-is or min-max across cohort |
| `peak` | Threshold from train cpu_scaled | Binary {0,1} — no scaling |

**Rule:** All normalisation statistics computed on **train period only**, frozen before validation/inference.

---

## Sequence construction (unchanged logic)

Reuse `create_residual_sequences()` with `features = [
    "residual_scaled",
    "prophet_yhat_scaled",
    "res_roll_std_12",
    "cpu_std",
    "is_peak_p90",
]`.

Per-container chronological sliding windows; `input_window=96`, `forecast_horizon=96`.

---

## Inference protocol alignment

At Day-1 forecast origin (matches frozen Hybrid evaluation):

1. Prophet forecast for validation timestamps already computed.
2. Input window = last 96 **historical** residuals ending at train/val boundary (or last observed residual before horizon).
3. For each step in window:
   - `yhat` from Prophet in-sample prediction on known past
   - `vol_12` from causal rolling over known past residuals
   - `peak` from known past `cpu_scaled` vs frozen train P90
4. GRU predicts 96 future **residuals**; add to Prophet forecast.

**No future CPU or future residuals** enter the input window.

---

## Minimal ablation ladder (future experiments)

| Variant | Shape | Purpose |
|---------|-------|---------|
| v1 baseline | 96×1 | Replicate current |
| v2a | 96×2 | + `prophet_yhat_scaled` only |
| v2b | 96×3 | + `res_roll_std_12` |
| v2c | 96×4 | + `cpu_std` |
| v2-full | 96×5 | + `is_peak_p90` |

Incrementally attribute any gain to specific context type.

---

## Architecture note

GRU input shape changes from `(96, 1)` to `(96, 5)` — same depth/width unless parameter budget study desired. Parameter count in first GRU layer scales linearly with `n_features` (negligible vs hidden units).

---

## Alternative considered and rejected

| Alternative | Why rejected |
|-------------|--------------|
| 96×20+ full feature suite | Redundancy, overfitting (RLLA: weak signal) |
| Separate context vector (1×k) + 96×1 residual | Breaks per-timestep alignment of level/volatility |
| Raw CPU multivariate (mem, net) | Data missing |
| Longer window (192+) with context | TMA: not supported for Hybrid residual path |
