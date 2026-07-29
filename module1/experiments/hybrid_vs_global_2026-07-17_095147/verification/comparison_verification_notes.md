# Hybrid vs Global GRU — Comparison Verification Notes

**Date:** 2026-07-17 (regenerated after Global baseline correction)  
**Phase:** 6 — Methodology Comparison (Task 4)  
**Experiment identifier:** `hybrid_vs_global_2026-07-17_095147`  
**Verification script:** `scripts/verify_hybrid_vs_global_comparison.py`

---

## Verification Summary

```
=== Hybrid vs Global GRU Comparison Verification ===
Selected containers : 100 (cohort design)
Evaluated containers: 99
Skipped containers  : 1 (c_14674)
Delta convention    : global_minus_hybrid
Comparison integrity: PASS
All checks          : PASS
```

| Check | Result |
|-------|--------|
| Both frozen datasets contain same 99 container IDs | **PASS** |
| `c_14674` absent from both datasets | **PASS** |
| No duplicate `container_id` after merge | **PASS** |
| Deltas = Global − Hybrid (MAE, RMSE, MAPE) | **PASS** |
| No NaN in comparison outputs | **PASS** |
| `comparison_table.csv` aggregates match frozen `evaluation_summary.csv` | **PASS** |
| Delta summary matches `phase6_comparison.json` | **PASS** |
| Saved artifacts match frozen-source recompute | **PASS** |
| `comparison_metadata.json` provenance fields | **PASS** |

**Full stdout log:** `verification/comparison_verification_run.log`

---

## Frozen References (Read-Only)

| Methodology | Reference |
|-------------|-----------|
| Hybrid Prophet + GRU | `experiments/baseline_reference_2026-07-14/` |
| Global GRU v1 (official) | `experiments/global_gru_baseline_2026-07-17_121748/` |
| Global GRU v1 (superseded) | `experiments/global_gru_reference_2026-07-17/` |

Comparison regenerated 2026-07-17 after Global baseline correction (Phase B). Hybrid reference unchanged.

---

## Primary Results Verified (Day 1, real CPU %)

| Methodology | MAE (mean) | RMSE (mean) | MAPE (mean) |
|-------------|------------|-------------|-------------|
| Hybrid Prophet + GRU | 1.7459 | 2.3878 | 111.9403 |
| Global GRU v1 | 1.9242 | 2.6105 | 116.5026 |
| **Δ (Global − Hybrid)** | **+0.1784** | **+0.2227** | **+4.5623** |

## Per-Container MAE Outcome

| Outcome | Count |
|---------|-------|
| Global better (lower MAE) | 32 |
| Global worse (higher MAE) | 67 |

---

## Gate Status

Comparison verification **PASS**. Artifacts reflect the corrected Global baseline (`patience=10`).

**Note:** MAE and RMSE are primary accuracy indicators; MAPE is supporting evidence only.
