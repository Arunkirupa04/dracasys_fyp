# Experimental Design

## Protocol identity

- **Name:** Global GRU Temporal Context Experiment (GGTCE)
- **Version:** `ggtce_v1.0`
- **Type:** Controlled input-window ablation (Global path only)

## Variants

| Variant | Input window | Output | Input shape | Calendar span (input) |
|---------|--------------|--------|-------------|------------------------|
| **G96** | 96 | 96 | (96, 3) | 1 day |
| **G192** | 192 | 96 | (192, 3) | 2 days |
| **G288** | 288 | 96 | (288, 3) | 3 days |
| G384* | 384 | 96 | (384, 3) | 4 days |

\*G384 — **discussion / sensitivity only**; not recommended as primary arm (see window_analysis.md).

## Isolation rules

1. New experiment directory only: `experiments/ggtce_{timestamp}/`
2. New module namespace recommended: `utils/ggtce/` (future — not created in protocol stage)
3. Do **not** modify `utils/global_config.py` locked constants in place — pass `input_window` as parameter
4. Do **not** overwrite `experiments/global_gru_baseline_2026-07-17_121748/`
5. Do **not** retrain or alter Hybrid, HCERL, CSRLE, LFHE, TMA artifacts

## Training protocol

1. Load frozen `train_df`, `selected_containers`, `scalers`
2. For each variant **independently**:
   - Build sequences with variant `input_window`
   - Train fresh Global GRU (same architecture builder with new `input_shape`)
   - Save model + `training_metadata.json` under `variants/g{window}/`
3. Log sequence counts; **fail fast** if any container produces zero sequences

## Inference protocol

At Day-1 forecast origin for each container:

- Input = last **`input_window`** scaled CPU feature vectors from **train period**
- Predict next 96 scaled CPU steps
- Inverse-transform to real CPU %
- Compare to validation actuals (first 96 steps)

**Critical:** Inference must use the **same window length as training** for each variant. Update from G96 spec language that says "last 96 steps" — GGTCE generalises to "last `input_window` steps."

## Analysis protocol

1. **Primary:** Cohort Day-1 MAE / RMSE / MAPE (paired vs G96)
2. **Secondary:** TMA-linked subgroup analysis (memory tertiles)
3. **TMA linkage:** Correlate per-container Δ MAE with TMA `capture_96`, `lag192_acf`, integrated autocorrelation time
4. **Parsimony:** Recommend shortest window within practical significance tolerance

## Ablation comparisons

| Transition | Tests |
|------------|-------|
| G96 → G192 | Value of second day of context |
| G192 → G288 | Value of third day / diminishing returns |
| G96 → G288 | Total long-context benefit |

## Optional sensitivity (not primary)

- G384 if reviewers request — document as under-powered sequence count
- Peak-subset Day-1 metrics (borrow peak thresholds from peak-aware experiment — read-only)
