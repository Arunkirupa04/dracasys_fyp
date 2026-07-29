# Baseline Freeze Statement

**Phase:** 1 — Freeze the Baseline  
**Date:** 2026-07-14  
**Status:** COMPLETE

---

## Control Model

The finalized baseline for all Peak-Aware Hybrid research is:

**Hybrid Prophet + GRU** (`hybrid_prophet_gru_v1`)

- Input window: **96** timesteps (24 hours)
- Forecast horizon: **96** timesteps (Day 1 primary)
- Prophet: daily seasonality on; weekly seasonality off
- GRU: residual learner on globally normalized Prophet residuals

---

## Immutable Reference Location

```
experiments/baseline_reference_2026-07-14/
├── models/           hybrid_gru.keras, residual_stats.pkl
├── evaluation/       evaluation_df.csv, evaluation_summary.csv
├── config/           baseline_metadata.json
├── verification/     verification_notes.md
└── FREEZE_STATEMENT.md
```

This folder is the **permanent control reference**. Peak-Aware experiments must not overwrite it.

---

## Control Metrics (Day 1, 99 containers, real CPU %)

| Metric | Mean | Std |
|--------|------|-----|
| MAE | 1.745882 | 2.486804 |
| RMSE | 2.387813 | 3.219141 |
| MAPE | 111.940318 | 615.299100 |

- **Evaluated:** 99 containers  
- **Skipped:** `c_14674` (missing from frozen train/val data)

---

## Frozen Components

The following must remain unchanged during Peak-Aware research:

- `data/` preprocessing artifacts
- `utils/hybrid_*.py` and `utils/sequence_utils.py`
- `notebooks/hybrid_model.ipynb` (baseline cells)
- `notebooks/preprocessing.ipynb`
- `models/hybrid_gru.keras` and `models/residual_stats.pkl` (production path)
- `docs/hybrid-prophet-gru.md`

Prophet models are **not persisted** — refit per container at inference.

---

## Peak-Aware Development Policy

- Implement only in **parallel modules** (e.g. `utils/peak_*.py`, `utils/hybrid_training_peak_aware.py`)
- Save Peak-Aware models only under `experiments/peak_aware_*`
- Never overwrite `models/` or `baseline_reference_2026-07-14/`

---

## Verification

All five baseline verification scripts passed on 2026-07-14. See `verification/verification_notes.md`.

---

## Next Phase

**Phase 2 — Peak Exploration** (no model training)
