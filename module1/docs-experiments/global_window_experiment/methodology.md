# Methodology — Frozen Global GRU Baseline

Summary of `global_gru_v1` from `docs/global-gru-baseline/` and `utils/global_config.py`. **All items below remain frozen in GGTCE except `input_window`.**

## Preprocessing (frozen)

| Step | Specification |
|------|---------------|
| Source | Alibaba `container_usage.csv` → `data/train_df.parquet`, `val_df.parquet` |
| Resampling | 15-minute mean per container |
| Interpolation | Per-container linear on CPU |
| Scaling | MinMaxScaler [0,1] fit on **train portion only** per container |
| Target | `cpu_scaled` |
| Static features | `cpu_mean`, `cpu_std` (train-computed, broadcast) |
| Split | Per-container 80/20 chronological |
| Cohort | `data/selected_containers.npy` (99 evaluable) |

## Sequence generation (variable: `input_window` only)

- Builder: `create_longterm_sequences()` in `utils/sequence_utils.py`
- Per-container chronological sliding windows
- **Input:** last `input_window` steps of `(cpu_scaled, cpu_mean, cpu_std)`
- **Target:** next 96 steps of `cpu_scaled`
- Constraint: `max_sequences = n_train - input_window - 96 + 1`

## Architecture (frozen)

```
GRU(128, return_sequences=True)
→ Dropout(0.2)
→ GRU(64)
→ Dense(64, relu)
→ Dense(96)
```

## Training (frozen)

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (default lr 0.001) |
| Loss | MSE |
| Metrics | MAE |
| Epochs max | 50 |
| Batch size | 256 |
| Shuffle | False |
| Early stopping | monitor=`val_loss`, patience=10, restore_best_weights=True |
| Internal split | Chronological 80/20 on **sequences** (early stopping only) |
| Random seed | 42 (`set_random_seeds`) |
| Training data | **Train period only** (`train_df` filtered to cohort) |

## Evaluation (frozen)

| Item | Specification |
|------|---------------|
| Horizon | Day-1 only (96 steps = 24 h) |
| Protocol | Same as Hybrid Day-1 (`utils/global_evaluation.py` → shared `hybrid_evaluation` metrics) |
| Metrics | MAE, RMSE, MAPE (real CPU % after inverse MinMax) |
| MAPE epsilon | 0.01 pp |
| Inference input | Last `input_window` train steps (not fixed at 96 in protocol — scales with variant) |
| Peak metrics | **Not** in Global baseline (optional exploratory add-on — see evaluation_protocol) |

## What GGTCE changes

| Component | G96 | G192 | G288 |
|-----------|-----|------|------|
| `input_window` | 96 | 192 | 288 |
| GRU `input_shape` | (96, 3) | (192, 3) | (288, 3) |
| First-layer GRU params | baseline | ~2× input | ~3× input |
| Everything else | **Identical** | **Identical** | **Identical** |

## What must NOT change

- Features (`cpu_scaled`, `cpu_mean`, `cpu_std`)
- Output horizon (96)
- Architecture depth/width
- Optimizer, loss, epochs, batch size, early stopping, seed
- Preprocessing artifacts
- Validation cohort and Day-1 evaluation procedure

## Control reference

Reload and compare against frozen G96 artifacts:

`experiments/global_gru_baseline_2026-07-17_121748/`

G96 in GGTCE should **replicate** this configuration (not Hybrid baseline).
