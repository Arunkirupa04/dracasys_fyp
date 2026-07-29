# CSRLE Results

**Authoritative experiment:** `experiments/synthetic_residual_learnability_2026-07-25_164307/`

---

## Stage 2 — Synthetic GRU (COMPLETE)

**Artifacts:** `stage2_synthetic_gru/`  
**Report:** [stage2_report.md](stage2_report.md)

### Residual learning (within-trajectory, Day-1, cohort mean)

| Condition | Pearson r | R² | Std ratio | frac r>0 |
|-----------|-----------|-----|-----------|----------|
| A (real) | 0.016 | −0.341 | 0.050 | 56% |
| B0 | 0.084 | −0.332 | 0.073 | 71% |
| B1 | 0.065 | −0.335 | 0.052 | 63% |
| C0 | 0.049 | −0.348 | 0.043 | 61% |
| C1 | 0.035 | −0.331 | 0.044 | 56% |

### Primary controlled comparisons

| Pair | Metric | Mean Δ (B−C) | 95% CI | Direction |
|------|--------|--------------|--------|-----------|
| B0 vs C0 | Pearson r | +0.035 | [0.004, 0.067] | B0 higher |
| B0 vs C0 | Std ratio | +0.030 | [0.025, 0.036] | B0 higher |
| B1 vs C1 | Pearson r | +0.030 | [0.004, 0.057] | B1 higher |

### CPU Day-1 MAE (Hybrid − Prophet)

| Condition | Δ MAE |
|-----------|-------|
| A | ~0 |
| B0 | **−0.006** |
| B1 | +0.003 |
| C0 | +0.001 |
| C1 | +0.003 |

### Baselines (B0, residual MAE / r)

| Method | MAE | r |
|--------|-----|---|
| GRU | 0.094 | 0.084 |
| Zero | 0.094 | — |
| Ridge | 0.093 | **0.158** |

---

## Stage 1 — Condition A Hybrid reproduction (PASS)

**Authoritative:** `stage1_control_reproduction/evaluation/reproduction_summary.json`

| Metric | CSRLE new model | Frozen baseline | Δ |
|--------|-----------------|-----------------|---|
| Day-1 MAE mean | 1.745 | 1.746 | −0.001 |
| Day-1 RMSE mean | 2.386 | 2.388 | −0.002 |
| Rank corr (per-container MAE) | — | — | 0.99998 |

Extended residual metrics (A): r ≈ 0.016, R² ≈ −0.341, std ratio ≈ 0.050.

---

## Pre-GRU v2 (PASS)

**Authoritative:** `synthetic_validation/pre_gru_summary.json`

| Gate | B0 | B1 | C0 | C1 |
|------|:--:|:--:|:--:|:--:|
| Generator | PASS | PASS | PASS | PASS |
| Clip | PASS | PASS | PASS | PASS |
| Retention | PASS | PASS | — | — |

α = 1 on 90/99 containers; 8 containers α = 0 (near-idle).

---

## v1 audit (preserved)

`experiments/synthetic_residual_learnability_2026-07-25_163013/` — clip gate FAILED.
