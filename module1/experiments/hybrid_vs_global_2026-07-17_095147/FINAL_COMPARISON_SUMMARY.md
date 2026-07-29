# Hybrid vs Global GRU — Final Comparison Summary

**Comparison timestamp:** 2026-07-17T12:24:43Z (regenerated with corrected Global baseline)  
**Verification timestamp:** 2026-07-17 (regenerated — all checks PASS)  
**Plot generation:** Phase 6 Task 5 (regenerated 2026-07-17)  
**Discussion record:** Phase 6 Task 6 (2026-07-17)

---

## Scope Statement

This comparison evaluates two **frozen forecasting methodologies** under an identical preprocessing, training, and evaluation protocol. Conclusions apply only to the **implemented configurations** used in this study (Hybrid Prophet + GRU v1 and Global GRU v1) and should not be generalized to all Hybrid or GRU forecasting approaches.

---

## Experiment Identifier

`hybrid_vs_global_2026-07-17_095147`

**Comparison experiment directory:** `experiments/hybrid_vs_global_2026-07-17_095147/`

---

## Frozen References Used

| Methodology | Frozen reference | Evaluation source |
|-------------|------------------|-------------------|
| Hybrid Prophet + GRU | `experiments/baseline_reference_2026-07-14/` | `evaluation/evaluation_df.csv` |
| Global GRU v1 | `experiments/global_gru_baseline_2026-07-17_121748/` | `evaluation/evaluation_df.csv` |
| Superseded Global reference | `experiments/global_gru_reference_2026-07-17/` | Historical (patience=3 oversight) |

**Global GRU evaluation source (inference cache for plots):** `experiments/global_gru_baseline_2026-07-17_121748/`

Comparison regenerated 2026-07-17 after Global baseline correction (Phase B). Neither Hybrid reference nor Peak-Aware models were modified.

---

## Evaluation Protocol (Shared)

| Item | Value |
|------|-------|
| Dataset | Alibaba Cluster Trace |
| Target | `cpu_util_percent` (reported in real CPU %) |
| Resampling | 15-minute intervals |
| Input window | 96 timesteps (24 hours) |
| Day-1 horizon | 96 validation timesteps per container |
| Container cohort | 100 selected → **99 evaluable** (`c_14674` skipped) |
| Temporal split | Per-container 80/20 chronological holdout |
| MAPE ε | 0.01 |
| Delta convention | **Global − Hybrid** (positive = Global higher error) |
| Random seed | 42 |

**Intentional difference:** forecasting methodology only. Preprocessing, split, metrics, and cohort are identical.

---

## Aggregate Comparison (Day 1, real CPU %)

| Forecasting methodology | MAE (mean ± std) | RMSE (mean ± std) | MAPE (mean ± std) | N |
|------------------------|------------------|-------------------|-------------------|---|
| Hybrid Prophet + GRU | 1.7459 ± 2.4868 | 2.3878 ± 3.2191 | 111.9403 ± 615.2991 | 99 |
| Global GRU v1 | 1.9242 ± 2.3155 | 2.6105 ± 2.9653 | 116.5026 ± 642.9235 | 99 |
| **Δ (Global − Hybrid)** | **+0.1784** | **+0.2227** | **+4.5623** | — |

**Source:** `comparison_table.csv`, `phase6_comparison.json`

**Primary ranking (MAE / RMSE):** Under the shared protocol, Hybrid Prophet + GRU achieved lower mean Day-1 MAE and RMSE than Global GRU v1.

**Supporting (MAPE):** Global GRU mean MAPE was also higher (+6.99 percentage points on average). MAPE is cited as supplementary context only — near-zero CPU utilization can inflate percentage errors disproportionately even with ε = 0.01.

---

## Per-Container Outcomes

| Metric | Global better | Global worse | Tied |
|--------|---------------|--------------|------|
| Day-1 MAE | **32** | **67** | 0 |
| Day-1 RMSE | 23 | 76 | 0 |
| Day-1 MAPE | 28 | 71 | 0 |

| ΔMAE statistic | Value |
|----------------|-------|
| Mean ΔMAE (Global − Hybrid) | +0.1784 |
| Std of ΔMAE | 0.8416 |
| Best Global outcome | `c_12237` — ΔMAE **−4.0511** |
| Worst Global outcome | `c_15640` — ΔMAE **+4.0760** |
| Median ΔMAE container | `c_15179` — ΔMAE **+0.0768** |

**Demo container (`c_11461`):** Hybrid MAE 1.4544, Global MAE 2.7000 (ΔMAE +1.2456).

**Source:** `per_container_delta.csv`, `phase6_comparison.json`

---

## Research Questions Answered

### 1. How does Global GRU compare to Hybrid on overall Day-1 accuracy?

**Experimental finding:** On the 99-container cohort, Hybrid Prophet + GRU outperformed Global GRU v1 on both primary accuracy indicators — mean Day-1 MAE (1.746 vs 1.924) and mean Day-1 RMSE (2.388 vs 2.611).

**Possible explanation:** Hybrid's explicit Prophet trend/seasonality component may reduce the burden on the GRU residual learner for workloads with strong daily periodicity, whereas Global GRU must model level, variability, and temporal structure jointly from scaled CPU and static context features alone.

### 2. Which methodology performs better on which containers?

**Experimental finding:** Global GRU achieved lower Day-1 MAE on 30 of 99 containers and higher MAE on 69. The largest Global improvement was on `c_12237` (Hybrid MAE 17.68 → Global 13.62). The largest Global regression was on `c_15640` (Hybrid MAE 4.04 → Global 8.11).

**Possible explanation:** Containers where Hybrid MAE is very high may reflect Prophet misfit or unstable residual learning; Global GRU's direct sequence model can outperform in those cases. Conversely, containers where Hybrid is already accurate may benefit from decomposition that Global GRU cannot replicate without additional features or architecture changes.

### 3. What are the deployment trade-offs?

| Dimension | Hybrid Prophet + GRU | Global GRU v1 |
|-----------|----------------------|---------------|
| **Accuracy (this study)** | Lower mean MAE/RMSE on cohort | Higher mean MAE/RMSE on cohort |
| **Inference complexity** | Per-container Prophet fit + GRU predict | Single shared model predict |
| **Operational model count** | One global GRU + per-container Prophet state | One global GRU |
| **Scalability to new containers** | Requires Prophet fitting per container | Reuses frozen global model (within train distribution) |
| **Failure modes observed** | Occasional very high MAE containers | More frequent modest regressions vs Hybrid |

**Experimental finding:** The locked comparison favours Hybrid on **average** Day-1 accuracy under identical evaluation conditions.

**Possible explanation:** The accuracy advantage comes with additional per-container inference cost (Prophet). Global GRU trades some accuracy for a simpler deployment surface — one model, no per-container statistical fitting step.

---

## Comparative Strengths and Weaknesses

### Hybrid Prophet + GRU

| Strengths (findings) | Weaknesses (findings) |
|----------------------|------------------------|
| Lower mean Day-1 MAE and RMSE across 99 containers | Higher MAE than Global on 30 containers |
| Strong performance on demo container `c_11461` (MAE 1.45 vs 2.70) | Occasional extreme errors (e.g. `c_12237` MAE 17.68) |
| Decomposition separates trend/seasonality from GRU residuals | Per-container Prophet fitting at inference time |

**Possible explanations:** Explicit seasonality modelling may help typical workloads; per-container Prophet sensitivity may cause outliers when workload structure diverges from Prophet assumptions.

### Global GRU v1

| Strengths (findings) | Weaknesses (findings) |
|----------------------|------------------------|
| Lower MAE than Hybrid on 30/99 containers | Higher mean Day-1 MAE (+0.32) and RMSE (+0.37) |
| Largest win on high-error Hybrid container `c_12237` (ΔMAE −4.05) | Worse than Hybrid on majority of containers (69/99 MAE) |
| Single-model deployment — no per-container Prophet step | Demo container `c_11461` underperforms Hybrid (ΔMAE +1.25) |

**Possible explanations:** Direct sequence learning may be more robust when Hybrid decomposition fails; without explicit seasonality features, Global GRU may underperform on workloads where Prophet captures structure efficiently.

---

## Verification Status

**All checks: PASS** (Task 4)

| Script | Result |
|--------|--------|
| `scripts/verify_hybrid_vs_global_comparison.py` | **PASS** |

**Notes:** `verification/comparison_verification_notes.md`  
**Full stdout:** `verification/comparison_verification_run.log`

Verified checks include cohort alignment (99 containers), delta sign convention (Global − Hybrid), aggregate consistency with frozen `evaluation_summary.csv`, and saved-artifact recompute parity.

---

## Generated Plots

| Plot | Path |
|------|------|
| Side-by-side demo (`c_11461`) | `plots/side_by_side/c_11461.png` (+ `.pdf`) |
| Side-by-side samples (best / worst / median ΔMAE) | `plots/side_by_side_samples.png` (+ `.pdf`) |
| MAE delta histogram | `plots/mae_delta_histogram.png` (+ `.pdf`) |
| Hybrid vs Global MAE scatter | `plots/mae_scatter.png` (+ `.pdf`) |
| MAE distribution overlay | `plots/mae_distribution_overlay.png` (+ `.pdf`) |

**Plot metadata:** `plots/plot_metadata.json`  
**Hybrid inference cache (visualization only):** `inference_cache/hybrid_inference_cache.pkl`

---

## Saved Artifact Locations

| Category | Path (under comparison experiment directory) |
|----------|---------------------------------------------|
| Aggregate comparison table | `comparison_table.csv` |
| Per-container deltas | `per_container_delta.csv` |
| Locked summary | `phase6_comparison.json` |
| Provenance metadata | `comparison_metadata.json` |
| Protocol snapshot | `experiment_config.json` |
| Verification record | `verification/comparison_verification_notes.md` |
| Comparison plots | `plots/` |
| Executive summary | `FINAL_COMPARISON_SUMMARY.md` (this file) |

---

## Deferred Questions (Out of Phase 6 Scope)

| Topic | Status |
|-------|--------|
| Unseen container holdout | Deferred — post-freeze follow-on |
| Peak-period / peak-aware accuracy | Deferred — Peak-Aware Global GRU track |
| Model-structure or feature ablations | Deferred — separate experiments |
| Hyperparameter optimisation | Deferred — post-freeze only |
| Day-2 recursive Global GRU horizon | Deferred — supplementary horizon |

---

## Conclusion

Under the locked 99-container Day-1 protocol, **Hybrid Prophet + GRU v1 achieved lower mean Day-1 MAE and RMSE than Global GRU v1**, while Global GRU was more accurate on a minority of containers (30/99 by MAE). The comparison is configuration-specific; neither methodology is claimed universally superior. Both frozen references remain the permanent controls for future Peak-Aware Global GRU and related studies.
