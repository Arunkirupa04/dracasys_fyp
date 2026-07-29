# Stage 3 — Evaluation Record (Peak-Aware Global GRU)

[← Implementation Record](02-implementation-record.md) · [Evaluation Plan](stage-03-evaluation-plan.md) · [Study Overview](README.md)

**Status:** Complete  
**Locked date:** 2026-07-17  
**Official treatment run:** `experiments/peak_aware_global_validation_vs1_2026-07-17_170939/`  
**Stage 3 lock:** `experiments/peak_aware_global_validation_vs1_2026-07-17_170939/phase3_evaluation.json`  
**Control reference:** `experiments/global_gru_baseline_2026-07-17_121748/`  
**Superseded control (historical):** `experiments/global_gru_reference_2026-07-17/`  
**Baseline promotion record:** `experiments/global_gru_baseline_2026-07-17_121748/official_baseline_promotion.json`  
**Verification:** `verification/evaluation_verification.json` — **10/10 PASS** (regenerated 2026-07-17 after baseline correction)

---

## 1. Objective

Execute a fair paired comparison between the frozen Global GRU baseline and the promoted Peak-Aware Global GRU treatment. Report overall Day-1 metrics plus peak- and non-peak-subset metrics under the locked P90 + λ=5 methodology — without modifying frozen baseline code or retraining models during evaluation.

---

## 2. Official Treatment Promotion

The original Stage 2 primary run (`peak_aware_global_2026-07-17_164737`, EarlyStopping patience=3) is archived as **Historical – optimization-limited**. The VS-1 validation study showed premature stopping limited performance, not the peak-aware methodology itself.

| Run | Patience | Epochs | Mean Day-1 MAE | Role |
|-----|----------|--------|----------------|------|
| `_164737` | 3 | 4 | 3.11 | Archived |
| `_170939` (VS-1) | 10 | 29 (best 19) | 2.17 | **Official treatment** |
| Official Global baseline | 10 | 35 (best 25) | 1.92 | Control |
| Superseded Global baseline | 3 | 9 | 2.06 | Historical (patience=3 oversight) |

Promotion record: `official_treatment_promotion.json`  
Global baseline promotion: `experiments/global_gru_baseline_2026-07-17_121748/official_baseline_promotion.json`  
Stage 1 design lock (λ, P90, architecture, evaluation protocol) unchanged; only training stopping patience amended for the official model. Stage 3 comparison artifacts were **regenerated** against the corrected Global baseline (Phase B, 2026-07-17) without retraining either model.

---

## 3. Evaluation Workflow

```
Official Global baseline (global_gru_baseline_2026-07-17_121748)
        ↓
Task 2: Treatment primary eval (evaluate_selected_containers)
        ↓
Task 3: Control re-run + reproducibility gate + peak-subset metrics
        ↓
Task 4: Comparison tables + plots (saved artifacts only)
        ↓
Task 5: 10-check evaluation verification
        ↓
Task 6: Stage 3 lock (this record)
        ↓
Phase B (2026-07-17): Regenerated comparison vs corrected Global baseline
```

Both models evaluated through identical `evaluate_selected_containers()` / `run_global_inference()` with the same 99-container cohort, train-fitted P90 thresholds, and Day-1 horizon (96 steps).

---

## 4. Task Summary (Tasks 1–6)

| Task | Focus | Status |
|------|-------|--------|
| 1 | Evaluation workspace + Global peak-evaluation adapters | **Complete** |
| 2 | Treatment primary evaluation (99 containers) | **Complete** |
| 3 | Peak-subset metrics + control reproducibility gate | **Complete** |
| 4 | Comparison tables + plots | **Complete** |
| 5 | Evaluation verification (10 checks) | **Complete — 10/10 PASS** |
| 6 | Lock evaluation record | **Complete** |

**Key scripts:**

| Script | Role |
|--------|------|
| `scripts/run_peak_aware_global_evaluation.py` | Task 2 treatment eval |
| `scripts/run_peak_aware_global_peak_subset_evaluation.py` | Task 3 peak-subset + control gate |
| `scripts/build_peak_aware_global_comparison.py` | Task 4 comparison builder |
| `scripts/verify_peak_aware_global_evaluation.py` | Task 5 verification gate |

---

## 5. Verification Summary

Evaluation verification (`verification/evaluation_verification.json`) passed all 10 checks:

1. Official treatment path (VS-1 promoted; archived not active source)
2. Control reproducibility gate PASS
3. Cohort consistency (99 eval, `c_14674` skipped, 0 failures)
4. Metric consistency across summary CSV, comparison table, and JSON
5. Peak-threshold consistency (99 train-fitted P90)
6. Peak-subset consistency (25.65% fraction; cache reproduction)
7. `per_container_comparison.csv` integrity (99 rows, full schema)
8. Plot integrity (6 PNG/PDF files)
9. Frozen evaluation path unmodified
10. Methodology audit (λ=5 only diff vs official Global baseline; both use patience=10)

**Non-blocking warnings:** Training metadata retains archived run as VS-1 lineage reference; Tier 1 vs Tier 2 divergence requires joint Stage 4 interpretation.

---

## 6. Key Numerical Findings

### Cohort-level comparison (official Global baseline vs official treatment)

*Regenerated 2026-07-17 after Global baseline correction (Phase B). Treatment model unchanged.*

| Tier | Metric | Baseline | Treatment | Δ (treatment − control) |
|------|--------|----------|-----------|-------------------------|
| Overall | MAE | 1.924 | 2.174 | **+0.249** |
| Overall | RMSE | 2.611 | 2.758 | **+0.147** |
| Overall | MAPE | 116.50 | 163.77 | **+47.27** |
| Peak subset | Peak MAE | 3.426 | 2.649 | **−0.777** |
| Peak subset | Peak RMSE | 6.559 | 5.316 | **−1.243** |
| Non-peak | Non-peak MAE | 1.406 | 2.010 | **+0.604** |
| Non-peak | Non-peak RMSE | 2.455 | 3.502 | **+1.047** |

Pooled peak timestep fraction: **25.65%** (within Hybrid Phase 2 feasibility range 17–31%).

### Per-container outcomes

| Item | Value |
|------|-------|
| Containers improved overall (ΔMAE < 0) | **23 / 99** |
| Containers worsened overall (ΔMAE > 0) | **76 / 99** |
| Containers improved on peak MAE | **89 / 99** |

---

## 7. Stage 3 Handoff Summary (Primary Stage 4 Input)

| Evaluation Tier | Result |
|-----------------|--------|
| **Overall** | Treatment MAE **+0.249** vs official Global GRU (2.174 vs 1.924); Tier 1 not improved |
| **Peak subset** | Treatment peak MAE **−0.777** (2.649 vs 3.426); **Tier 2 contribution achieved** |
| **Non-peak subset** | Treatment non-peak MAE **+0.604** (2.010 vs 1.406); non-peak degradation offsets peak gains |
| **Containers improved (overall)** | **23 / 99** |
| **Containers improved (peak)** | **89 / 99** |
| **Evaluation verification** | **10 / 10 PASS** (regenerated vs corrected baseline) |

**Primary Stage 4 artifact:** `evaluation/per_container_comparison.csv`

---

## 8. Final Stage 3 Conclusion

Under the locked fair-comparison protocol, Peak-Aware Global GRU training (λ=5, P90, timestep-weighted MSE) **improves accuracy on peak validation timesteps** but **degrades non-peak and overall Day-1 accuracy** relative to the official Global GRU baseline. Against the corrected baseline (patience=10), the overall gap widens (+0.249 MAE vs +0.111 under the superseded reference) because the stronger control reduces baseline error. Peak-subset gains remain substantial (−0.777 peak MAE). The VS-1 promoted model (EarlyStopping patience=10) represents the methodology under adequate optimization; the archived patience=3 run was optimization-limited.

Stage 3 is **complete and locked**. All evaluation artifacts are verified, reproducible, and immutable. Stage 4 may proceed with discussion, limitation framing, and architecture sensitivity synthesis using `per_container_comparison.csv` as the primary per-container input.

---

## 9. Locked Artifacts

| Artifact | Path |
|----------|------|
| Stage 3 lock JSON | `phase3_evaluation.json` |
| Per-container comparison | `evaluation/per_container_comparison.csv` |
| Cohort comparison table | `evaluation/comparison_table.csv` |
| Evaluation verification | `verification/evaluation_verification.json` |
| Control reproducibility gate | `evaluation/baseline_reproducibility_check.json` |
| Comparison plots | `evaluation/plots/` |

---

*Stage 3 locked 2026-07-17 — ready for Stage 4 Discussion*
