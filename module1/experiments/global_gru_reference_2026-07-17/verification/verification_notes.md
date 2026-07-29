# Global GRU v1 — Verification Notes

**Reference:** `experiments/global_gru_reference_2026-07-17/`  
**Training experiment:** `global_gru_baseline_2026-07-17_074743`  
**Evaluation experiment:** `global_gru_evaluation_2026-07-17_092120`

---

## Phase 4 — Implementation Verification (2026-07-17)

### Pre-Training Gate

| Script | Result | Notes |
|--------|--------|-------|
| `scripts/verify_global_gru_data_split.py` | **PASS** | 41,650 sequences from `global_train` only; 0 val-period target leaks |

| Metric | Value |
|--------|-------|
| Sequences (`global_train` only) | 41,650 |
| Val-period leaks (correct path) | 0 |
| Val-period leaks (legacy `concat`) | 15,120 (contrast only) |

### Hybrid Regression (Shared Utilities)

| Script | Result |
|--------|--------|
| `scripts/verify_hybrid_train_eval_split.py` | **PASS** |

`utils/hybrid_*.py`, frozen Hybrid models, and `experiments/baseline_reference_2026-07-14/` were not modified.

### Implementation Checks

| Check | Result |
|-------|--------|
| Phase 0.5 architecture (`Dropout(0.2)`) | **PASS** |
| Metadata includes `baseline_spec`, `random_seed=42` | **PASS** |
| Production model loads and runs inference | **PASS** |

---

## Phase 5 — Evaluation Verification (2026-07-17T09:28:10Z)

### Verification Suite

| Script | Result |
|--------|--------|
| `scripts/verify_global_gru_data_split.py` | **PASS** |
| `scripts/verify_global_gru_inference.py` | **PASS** |
| `scripts/verify_global_gru_evaluation.py` | **PASS** |
| `scripts/verify_global_gru_train_eval_split.py` | **PASS** |
| `scripts/verify_global_gru_methodology_parity.py` | **PASS** |

### Cohort Summary

```
Selected containers : 100
Evaluable containers: 99
Evaluated containers: 99
Skipped containers  : 1 (c_14674)
Failures            : 0
Evaluation integrity: PASS
```

### Evaluation Output Integrity

| Check | Result |
|-------|--------|
| Exactly 99 containers evaluated | **PASS** |
| Exactly 1 skipped (`c_14674`) | **PASS** |
| No duplicate container IDs | **PASS** |
| No NaN metric values | **PASS** |
| MAE, RMSE, MAPE non-negative | **PASS** |
| RMSE ≥ MAE per container | **PASS** |
| Saved artifacts match live rerun | **PASS** |

### Aggregate Metrics (Day 1, real CPU %)

| Metric | Mean | Std |
|--------|------|-----|
| MAE | 2.0627 | 2.4701 |
| RMSE | 2.7559 | 3.1039 |
| MAPE | 118.9320 | 659.0400 |

**Full log:** `experiments/global_gru_evaluation_2026-07-17_092120/verification/verification_run.log`

---

## Conclusion

Phase 4 implementation and Phase 5 evaluation verification are **complete**. The frozen reference evaluation artifacts are immutable and represent the official Global GRU v1 baseline evaluation for research comparison. Methodology comparison with Hybrid is deferred to Phase 6.
