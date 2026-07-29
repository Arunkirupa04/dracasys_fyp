# Methodology Gap Analysis

**Date:** July 2026  
**Compares:** Current notebook implementation vs. intended Module 1 research design  
**Reference:** `.cursor/skills/dracasys/SKILL.md`

---

## Compliance Summary

| Area | Intended Design | Compliance | Severity |
|------|-----------------|------------|----------|
| Preprocessing pipeline | 9-stage pipeline, 15min, per-container norm | Mostly aligned | Low–Medium |
| Prophet (Hybrid) | Trend + daily + **weekly** seasonality | Partial | Medium |
| Residual generation | `cpu_scaled − prophet_pred` | Aligned | — |
| Hybrid GRU | Learn nonlinear residuals | Aligned (window deviation) | Medium |
| Hybrid prediction | Prophet + GRU residual | Aligned | — |
| Sequence generation | 96 in / 96 out sliding windows | Hybrid: **288 in**; Global: 96 in | Medium–High |
| Train/validation split | Chronological, no leakage | Preprocessing ✓; Global GRU ✗ | **High** |
| Evaluation | MAE, RMSE, MAPE; fair comparison; unseen containers | Partial | **High** |

### Overall Verdict

| Architecture | Verdict |
|--------------|---------|
| **Preprocessing** | Largely correct |
| **Hybrid Prophet + GRU** | Conceptually correct; deviations in Prophet seasonality, input window, evaluation scope |
| **Global GRU** | Directionally correct but **train/val procedure violates time-series methodology** |
| **Cross-architecture comparison** | **Not yet valid** |

---

## 1. Preprocessing

### Intended

```
Raw CSV → timestamps → filter → order → resample(15min) → interpolate
→ per-container normalize → feature engineering → export
```

- Target: `cpu_util_percent` only
- Chronological 80/20 split before fitting scalers
- No fitting on future data

### Current — Aligned

- 15-minute resampling, interpolation, per-container MinMaxScaler (train-fit only)
- Summary stats from train portion only
- Chronological per-container 80/20 split
- Exports parquet files

### Deviations

| Issue | Detail | Severity |
|-------|--------|----------|
| Container filtering uses full timeline | Stats for selection computed on all raw data including later val period | Low–Medium |
| `scalers.pkl` not saved in notebook | Downstream notebooks depend on manually created artifact | Medium |
| Unused features | `hour`, `cpu_max`, `cpu_min` engineered but unused | Low |
| Workload typing unused | Stable/medium/spiky labels not used for selection or evaluation | Low |

**Data leakage:** Low at this stage.

---

## 2. Prophet Implementation

### Intended

Prophet models:

- Long-term trend
- Daily seasonality
- **Weekly seasonality**

### Current

```python
Prophet(
    daily_seasonality=True,
    weekly_seasonality=False   # disabled in all notebooks
)
```

### Aligned

- Per-container Prophet on training period (for holdout evaluation)
- Target: normalized CPU (`cpu_scaled`)
- Out-of-sample Prophet forecast for validation timestamps at inference

### Deviations

| Issue | Detail |
|-------|--------|
| **`weekly_seasonality=False`** | Direct deviation from Architecture 1 specification |
| In-sample residuals for GRU training | Prophet predicts points it was fit on — residuals may be slightly optimistic |
| Prophet not persisted | Refit every run; no versioned artifact |
| No documented hyperparameter tuning | Default Prophet settings used |

**Pragmatic justification:** ~6.4 days of train data per container — weekly seasonality is difficult to identify. If disabled intentionally, document this in the thesis.

---

## 3. Residual Generation

### Intended

```
residual = actual − prophet_forecast
Final forecast = prophet_forecast + gru_residual_prediction
```

### Current — Aligned

- Training: `residual = cpu_scaled - prophet_pred` on train period
- Global z-score: `(residual - res_mean) / res_std` using train-period residuals only
- Inference: recomputes train residuals; applies saved `res_mean`/`res_std`
- Final combination matches research formula

### Deviations

| Issue | Detail |
|-------|--------|
| Global residual scaling | Single mean/std across all 100 containers — may under-represent per-container scale |
| Day 2 recursive forecast | Uses predicted Day 1 residuals — error compounding beyond 96-step spec |

**Data leakage:** Low — statistics and Prophet use train period only during holdout evaluation.

---

## 4. GRU Implementation

### Intended

| Architecture | Role | Scope |
|--------------|------|-------|
| Hybrid GRU | Residual forecasting | Per hybrid pipeline |
| Global GRU | Direct CPU forecasting | Single model across containers |

### Current

**Hybrid GRU:**

- Input: `(288, 1)` — `residual_scaled`
- Output: 96 steps
- Saved to `models/hybrid_gru.keras`

**Global GRU:**

- Input: `(96, 3)` — `cpu_scaled`, `cpu_mean`, `cpu_std`
- Output: 96 steps
- **Not saved**

### Deviations

| Issue | Detail | Severity |
|-------|--------|----------|
| Hybrid input window = 288, not 96 | Spec default is 24h (96 steps) | Medium |
| Architectures not comparable | Different depth, width, features, input lengths | High |
| Variable naming | Hybrid GRU named `global_model` | Low |
| No random seeds | Non-reproducible training | Medium |
| Global GRU not persisted | Cannot reproduce Architecture 2 | Medium |

---

## 5. Hybrid Prediction

### Intended

- Final Forecast = Prophet + GRU Residual
- Horizon: 96 steps (24 hours)

### Current — Aligned

- Two-stage combination implemented correctly
- Day 1 = 96 steps (matches spec)
- Evaluation on preprocessing validation period (true temporal holdout)

### Deviations

| Issue | Detail |
|-------|--------|
| 2-day evaluation | Day 2 (steps 97–192) exceeds 96-step default |
| Single-container demo | One container (`c_11461`), not aggregated over 100 |

---

## 6. Sequence Generation

### Intended

- Sliding windows: 96 input → 96 output
- Chronological order preserved
- No shuffling

### Current

| Path | Input | Output | Source Data |
|------|-------|--------|-------------|
| Hybrid | **288** | 96 | Train period only |
| Global GRU | 96 | 96 | **Train + val concatenated** |

### Global GRU Sequence Analysis (100 containers)

| Sequence Type | Count | Percentage |
|---------------|-------|------------|
| Target entirely in train period | 42,172 | 73.4% |
| Target crosses train/val boundary | 9,500 | 16.5% |
| Target entirely in val period | 5,773 | 10.0% |

**26.6% of Global GRU sequences have targets in the validation period.**

With an 80/20 index split, approximately **21% of all sequences** (val-period targets) can enter GRU **training** — this is validation-target leakage.

---

## 7. Train/Validation Split

### Intended

- Chronological train/validation/test split
- Future data never used in training
- Identical splits for model comparison

### Current — Three Split Layers

```
Layer 1 (preprocessing):  Per-container 80/20 temporal split     ✓ Correct
Layer 2a (Hybrid GRU):    80/20 index split on train sequences   ~ Acceptable (early stopping)
Layer 2b (Global GRU):    80/20 index split on train+val seqs   ✗ Leakage
Layer 3 (Hybrid eval):    Preprocessing val period holdout       ✓ Correct
Layer 3 (Global eval):    Layer 2b val sequences                ✗ Not equivalent
```

| Split Aspect | Hybrid | Global GRU |
|--------------|--------|------------|
| Preprocessing temporal split | ✓ | ✓ |
| GRU training data scope | Train period only | Train + val periods |
| GRU val split method | Index 80/20 | Index 80/20 |
| Final forecast evaluation | Temporal holdout | Mixed-period sequences |
| Container holdout | None | None |
| Test set | None | None |

### Critical Leakage (Global GRU)

Concatenating `global_train` and `global_val`, building sequences over the full timeline, then splitting 80/20 by sequence index allows the model to **train on targets from the preprocessing validation period**.

---

## 8. Evaluation Methodology

### Intended

- Metrics: MAE, RMSE, **MAPE**
- Identical datasets and splits
- Evaluate on **unseen containers**
- Actual vs Predicted, residual, error distribution plots
- Fair, reproducible procedures

### Current

| Requirement | Hybrid | Global GRU |
|-------------|--------|------------|
| MAE | ✓ (1 container) | ✓ |
| RMSE | ✓ | ✓ |
| MAPE | ✗ | ✗ |
| Same evaluation containers | ✗ (`c_11461` vs `c_11307`) | ✗ |
| Same evaluation period | ✓ (temporal holdout) | ✗ (sequence split) |
| Aggregate over 100 containers | ✗ | Partial |
| Unseen container eval | ✗ | ✗ |
| Saved plots (PNG/PDF) | ✗ | ✗ |

### Reported Results (Not Directly Comparable)

| Model | Context | MAE | RMSE |
|-------|---------|-----|------|
| Hybrid Day 1 | Val holdout, 1 container | 1.47 | 2.03 |
| Hybrid Day 2 | Recursive | 1.85 | 2.45 |
| Global GRU | Scaled space | ~0.016 | — |
| Global GRU | Real CPU %, 1 container | ~1.99 | ~2.47 |

Global GRU scaled MAE (~0.016) is on MinMax [0,1] data — not comparable to Hybrid's real CPU % without consistent denormalization and identical evaluation windows.

---

## Issue Register

### Incorrect Implementations

1. **Global GRU trains on validation-period targets** — most serious
2. **Prophet `weekly_seasonality=False`** — contradicts Architecture 1 spec
3. **Hybrid 288-step input** — contradicts default 96-step config
4. **Architectures compared with different protocols** — undermines Research Objective 2
5. **Mislabeled metrics** — Global GRU notebook uses variable name `hybrid_mae` for Global GRU results

### Data Leakage Summary

| Location | Risk |
|----------|------|
| Preprocessing normalization | Low ✓ |
| Container filtering stats | Low–Medium |
| Hybrid Prophet + residuals | Low ✓ |
| Hybrid GRU sequence split | Low (early stopping only) |
| **Global GRU sequence build + split** | **High** |

### Shortcuts

- First 100 containers (`unique()[:100]`) — no stratification
- In-sample Prophet residuals for GRU training
- Single-container demo evaluation
- No random seeds or experiment logging
- Plots not exported to disk

### Missing Research Components

| Component | Status |
|-----------|--------|
| Unseen container evaluation | Not implemented |
| Fair Hybrid vs Global GRU comparison | Not implemented |
| MAPE metric | Not implemented |
| Weekly seasonality in Prophet | Disabled |
| Test set (train/val/test) | Not implemented |
| Stratified container selection | Not implemented |
| Multi-container aggregated evaluation | Not implemented |
| Reproducible experiment tracking | Not implemented |
| Modular pipeline | Not implemented |

### Cross-Architecture Inconsistencies

| Item | Hybrid | Global GRU |
|------|--------|------------|
| Input window | 288 | 96 |
| Features | 1 | 3 |
| Training data scope | Train period | Train + val |
| Final eval protocol | Temporal holdout | Sequence index split |
| Demo container | `c_11461` | `c_11307` |
| Model saved | Yes | No |

---

## Research Gap Coverage

| Research Gap (from spec) | Addressed? |
|----------------------------|------------|
| Gap 1: Generalization across containers | Explored via Global GRU; no unseen-container test |
| Gap 2: Statistical + deep learning hybrid | Yes — Prophet + GRU |
| Gap 3: Scalability & deployment practicality | Global GRU direction; not productionized |
| Gap 4: Irregular/missing data | Interpolation + resampling handled |
| Gap 5: Adaptive/online learning | Not implemented (planned extension) |

---

## Related Documents

- [Implementation Roadmap](implementation-roadmap.md) — prioritized action plan (fix → research → future)
- [Project Audit](project-audit.md) — file inventory and repository status
- [Data Pipeline Reference](data-pipeline-reference.md) — stage-by-stage data flow
- [Recommended Evaluation Protocol](recommended-evaluation-protocol.md) — how to fix gaps
