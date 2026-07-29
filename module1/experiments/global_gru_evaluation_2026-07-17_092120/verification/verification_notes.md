# Global GRU v1 — Phase 5 Verification Notes

**Date:** 2026-07-17  
**Phase:** 5 — Evaluation & Verification (Task 4)  
**Experiment identifier:** `global_gru_evaluation_2026-07-17_092120`  
**Evaluation model:** `models/global_gru.keras`  
**Frozen implementation reference:** `experiments/global_gru_reference_2026-07-17/`

---

## Verification Suite Summary

```
=== Global GRU Evaluation Verification ===
Selected containers : 100
Evaluable containers: 99
Evaluated containers: 99
Skipped containers  : 1 (c_14674)
Failures            : 0
Evaluation integrity: PASS
All checks          : PASS
```

| Script | Result | Notes |
|--------|--------|-------|
| `scripts/verify_global_gru_data_split.py` | **PASS** | 41,650 sequences from `global_train` only; 0 val-period target leaks |
| `scripts/verify_global_gru_inference.py` | **PASS** | `c_11461` manual vs module — arrays identical; MAE 2.7000, RMSE 3.4544, MAPE 25.8092 |
| `scripts/verify_global_gru_evaluation.py` | **PASS** | Saved artifacts + live rerun; evaluation integrity checks all pass |
| `scripts/verify_global_gru_train_eval_split.py` | **PASS** | Production model vs round-tripped load — 99 containers identical |
| `scripts/verify_global_gru_methodology_parity.py` | **PASS** | Cohort, schema, protocol constants aligned with Hybrid baseline |

**Suite run:** 2026-07-17T09:28:10Z  
**Full log:** `verification/verification_run.log`

---

## Data Integrity

| Check | Result |
|-------|--------|
| Sequences built from `global_train` only | **PASS** |
| Zero val-period target leaks | **PASS** |
| Legacy concat contrast shows expected leakage (15,120) | **PASS** |
| Notebook has no `pd.concat([global_train, global_val])` | **PASS** |

---

## Inference Integrity

| Check | Result |
|-------|--------|
| `run_global_inference()` matches manual temporal holdout | **PASS** |
| Demo container `c_11461` Day 1 arrays identical | **PASS** |
| Metrics recomputed from shared `hybrid_evaluation` functions | **PASS** |

---

## Evaluation Output Integrity

| Check | Result |
|-------|--------|
| Exactly 99 containers evaluated | **PASS** |
| Exactly 1 container skipped (`c_14674`) | **PASS** |
| No duplicate container IDs | **PASS** |
| No NaN metric values | **PASS** |
| MAE, RMSE, MAPE non-negative | **PASS** |
| RMSE ≥ MAE for every container | **PASS** |
| Saved `evaluation_df.csv` matches live rerun | **PASS** |
| Saved `evaluation_summary.csv` matches recomputed summary | **PASS** |

### Cohort aggregate metrics (live rerun)

| Metric | Mean | Std |
|--------|------|-----|
| MAE | 2.0627 | 2.4701 |
| RMSE | 2.7559 | 3.1039 |
| MAPE | 118.9320 | 659.0400 |

---

## Artifact / Load Integrity

| Check | Result |
|-------|--------|
| Production model save/load round-trip | **PASS** |
| Demo container inference (production vs loaded) | **PASS** |
| Full 99-container evaluation (production vs loaded) | **PASS** |

---

## Methodology Parity (Readiness — Not Comparison)

| Check | Result |
|-------|--------|
| `input_window = 96`, `forecast_horizon = 96` | **PASS** |
| `mape_epsilon = 0.01` (shared with Hybrid) | **PASS** |
| Identical 99 evaluated container IDs vs Hybrid baseline | **PASS** |
| `evaluation_df` schema matches Hybrid baseline | **PASS** |
| Hybrid control (`baseline_reference_2026-07-14/`) unmodified | **PASS** |

**Note:** Performance comparison deferred to Phase 6.

---

## Frozen Reference Status

Evaluation outputs in this experiment directory are **verified** but **not yet copied** to `experiments/global_gru_reference_2026-07-17/`. Artifact persistence to the frozen reference is Task 6 (post-verification gate satisfied).

---

## Conclusion

All five Phase 5 verification scripts **PASS**. The Global GRU v1 evaluation under the shared 99-container Day 1 protocol is verified and ready for plotting (Task 5) and artifact freeze (Task 6).
