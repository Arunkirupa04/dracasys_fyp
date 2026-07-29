# CSRLE Stage 2 — Final Report

**Date:** 2026-07-25  
**Authoritative artifacts:** `experiments/synthetic_residual_learnability_2026-07-25_164307/stage2_synthetic_gru/`  
**Orchestrator:** `scripts/run_csrle_stage2_synthetic_gru.py`

---

## 1. Artifacts created

| Path | Contents |
|------|----------|
| `stage2_synthetic_gru/b0_lc/` | Model, training, evaluation, diagnostics, plots |
| `stage2_synthetic_gru/b1_snar/` | Same |
| `stage2_synthetic_gru/c0_iid/` | Same |
| `stage2_synthetic_gru/c1_iid/` | Same |
| `stage2_synthetic_gru/comparisons/` | Cross-condition tables, paired bootstrap, α analysis |
| `stage2_synthetic_gru/plots/` | Cohort distribution, horizon, paired, training loss |
| `stage2_synthetic_gru/stage2_final_report.json` | Machine-readable summary |
| `utils/csrle/stage2_*.py` | Training, metrics, baselines, stats, plots modules |

Prior artifacts (v1, pre-GRU v2, Stage 1) were **not modified**.

---

## 2. Verification status

| Check | Status |
|-------|--------|
| Same 99 containers | PASS |
| Frozen `synthetic_config_frozen.json` unchanged | PASS |
| Independent GRU per condition (no weight sharing) | PASS |
| Condition A not retrained | PASS |
| Train-only scalers / residual normalization | PASS |
| Stage 1 / pre-GRU artifacts preserved | PASS |

---

## 3. Training summary

| Condition | Sequences | Epochs | Best epoch | Best val loss |
|-----------|-----------|--------|------------|---------------|
| B0 | 41,650 (33,320 / 8,330) | 12 | 2 | 1.705 |
| B1 | 41,650 | 14 | 4 | 1.674 |
| C0 | 41,650 | 13 | 3 | 1.687 |
| C1 | 41,650 | 13 | 3 | 1.688 |

Training losses were similar across conditions; no condition showed dramatically different convergence.

---

## 4. Overall residual metrics (within-trajectory, Day-1)

| Condition | Pearson r (mean) | R² (mean) | MAE | Std ratio | frac r>0 | frac r>0.30 |
|-----------|------------------|-----------|-----|-----------|----------|-------------|
| **A (real)** | 0.016 | −0.341 | 0.095 | 0.050 | 56% | 6% |
| **B0** | 0.084 | −0.332 | 0.094 | 0.073 | 71% | 14% |
| **B1** | 0.065 | −0.335 | 0.094 | 0.052 | 63% | 13% |
| **C0** | 0.049 | −0.348 | 0.093 | 0.043 | 61% | 8% |
| **C1** | 0.035 | −0.331 | 0.094 | 0.044 | 56% | 7% |

Source: Stage 1 extended metrics (A); `comparisons/all_conditions_residual_summary.csv` (synthetic).

---

## 5–8. Paired comparisons

### B0 vs C0 (full 99)

| Metric | Mean Δ (B0−C0) | 95% CI | frac B0 better |
|--------|----------------|--------|----------------|
| Pearson r | +0.035 | [0.004, 0.067] | 57% |
| R² | +0.015 | [0.001, 0.031] | 59% |
| Std ratio | +0.030 | [0.025, 0.036] | **94%** |

### B0 vs C0 (active injection α>0, n=91)

| Metric | Mean Δ | 95% CI |
|--------|--------|--------|
| Pearson r | +0.041 | [0.009, 0.073] |
| R² | +0.018 | [0.003, 0.034] |

### B1 vs C1 (full 99)

| Metric | Mean Δ (B1−C1) | 95% CI | frac B1 better |
|--------|----------------|--------|----------------|
| Pearson r | +0.030 | [0.004, 0.057] | 60% |
| R² | −0.004 | [−0.023, 0.014] | 58% |
| Std ratio | +0.008 | [0.005, 0.011] | 74% |

Active-injection B1 vs C1: Pearson r Δ +0.027, CI [0.001, 0.054].

---

## 9. Horizon-wise (cross-container correlation at h)

See per-condition `evaluation/cross_horizon_correlation.csv` and `plots/horizon_mae_by_condition.png`.

---

## 10. Horizon-band (within-trajectory Pearson r, cohort mean)

| Band | B0 | B1 | C0 | C1 |
|------|-----|-----|-----|-----|
| 1–4 | 0.151 | 0.091 | 0.045 | 0.140 |
| 5–12 | 0.099 | 0.073 | 0.065 | 0.054 |
| 13–24 | 0.150 | 0.065 | 0.121 | 0.145 |
| 25–48 | −0.006 | 0.067 | 0.031 | −0.009 |
| 49–96 | 0.128 | 0.085 | 0.062 | 0.047 |

B1 does **not** show a clean monotonic short→long horizon decay.

---

## 11. Within-trajectory

Primary metrics in `extended_metrics.csv` per condition; cohort summaries in `residual_summary.json`.

---

## 12. Synthetic component recovery (B0/B1)

| Condition | Ĝ vs R_B (r mean) | Ĝ vs N (r mean) |
|-----------|-------------------|-----------------|
| B0 | 0.084 | 0.173 |
| B1 | 0.065 | 0.171 |

Primary target R_B correlation matches residual Pearson r by construction. Supporting N correlation is higher but GRU is not trained on N directly.

---

## 13. Baselines (B0 cohort mean residual MAE / Pearson r)

| Predictor | MAE | Pearson r |
|-----------|-----|-----------|
| GRU | 0.094 | 0.084 |
| Zero | 0.094 | n/a (constant) |
| Persistence | 0.108 | n/a |
| **Ridge** | 0.093 | **0.158** |

Ridge achieves higher residual correlation than GRU on B0; GRU ≈ zero on MAE.

---

## 14. Prophet vs Hybrid CPU (Day-1 MAE mean)

| Condition | Prophet | Hybrid | Δ |
|-----------|---------|--------|---|
| A | 1.745 | 1.745 | ~0 |
| B0 | 1.773 | 1.767 | **−0.006** |
| B1 | 1.787 | 1.790 | +0.003 |
| C0 | 1.779 | 1.780 | +0.001 |
| C1 | 1.761 | 1.763 | +0.003 |

Residual learning translates to negligible CPU improvement except slight B0 gain.

---

## 15–16. Paired bootstrap & α=0

- 8 containers with α=0; B0 mean r on α=0 subset: −0.037 vs α>0: 0.095  
- Full cohort remains primary; active-injection analyses confirm same direction

---

## 17. Training behavior

Similar sequence counts and early stopping across conditions; no evidence that synthetic conditions changed learnability via training dynamics alone.

---

## 18. Main plots

- `plots/residual_pearson_r_distribution.{png,pdf}`
- `plots/residual_r2_distribution.{png,pdf}`
- `plots/residual_std_ratio_distribution.{png,pdf}`
- `plots/horizon_mae_by_condition.{png,pdf}`
- `plots/paired_b0_lc_vs_c0_iid_pearson_r.{png,pdf}`
- `plots/paired_b1_snar_vs_c1_iid_pearson_r.{png,pdf}`
- `plots/training_loss_comparison.{png,pdf}`
- Per-condition residual trajectories and synthetic recovery plots

Representative containers selected at cohort **median** residual Pearson r proximity.

---

## 19. Interpretation Q1–Q10

**Q1. Did the GRU learn B0 residual dynamics?**  
Partially. Mean within-trajectory Pearson r rose from 0.016 (A) to 0.084 (B0) and predicted/actual std ratio from 0.050 to 0.073, but R² remained negative (−0.332). Learning signal exists but is weak relative to injected structure.

**Q2. Was B0 learning materially stronger than C0?**  
Yes on several metrics, with modest effect sizes: Pearson r bootstrap CI excludes zero; std-ratio improvement is consistent (94% containers). Not decisive on MAE alone.

**Q3. Did the GRU learn B1 residual dynamics?**  
Weakly (r ≈ 0.065), between A and B0.

**Q4. Was B1 learning materially stronger than C1?**  
Marginally for Pearson r and std ratio; R² difference not significant.

**Q5. Did B1 performance decline with forecast horizon?**  
Not clearly. Horizon-band correlations do not monotonically decrease.

**Q6. Did the GRU recover meaningful residual variance in B0/B1 vs A?**  
Modest improvement: B0 std ratio 0.073 vs A 0.050; B1 0.052 ≈ A.

**Q7. Did improved residual learning translate to better CPU forecasts?**  
Minimal. B0 Hybrid MAE improved by 0.006 CPU points; other conditions slightly worse or flat.

**Q8. GRU vs zero/persistence/Ridge?**  
GRU ≈ zero on MAE; persistence worse. **Ridge beats GRU** on residual Pearson r (0.158 vs 0.084 on B0), suggesting forecastable linear structure the GRU under-exploits.

**Q9. Does the pipeline support capability claims?**  
Under B0, the fixed pipeline shows **limited but non-zero** capacity to exploit controlled structure—stronger than matched IID C0, weaker than Ridge. Not evidence of general effectiveness.

**Q10. B0 modest / A weak — data vs architecture?**  
Evidence favors **residual learnability and training objective limitations on real Alibaba residuals**, not complete architectural incapability—but the synthetic effect size is small and does not rescue R².

---

## 20. Documentation updated

See `docs/csrle/{README,methodology,results,discussion,implementation_log,reproducibility,validation_protocol}.md`.

---

## 21. Limitations

- R² remains negative across all conditions; correlation gains are modest  
- Ridge outperforms GRU on B0 — GRU/training may be the bottleneck  
- CPU-level gains negligible except tiny B0 improvement  
- α=0 containers dilute B0/B1 signal (8/99)  
- Prophet refit per container at inference is stochastic; GRU weights not seed-locked  

---

## 22. Unexpected findings

- Ridge residual correlation exceeds GRU on positive control B0  
- B1 horizon decay hypothesis not supported by band analysis  
- C1 residual r slightly below A despite synthetic setup  

---

## 23. Final experimental conclusion

The fixed Hybrid Prophet + GRU pipeline exhibits **detectable but limited** residual-learning capacity under deterministic synthetic B0-LC structure that survives Prophet, exceeding matched IID control C0 on correlation and variance-recovery metrics. It does **not** achieve strong R², does **not** clearly outperform linear Ridge, and does **not** materially improve Day-1 CPU forecasts except a small B0 delta. Condition A (real data) remains much weaker in absolute terms for residual shape learning.

---

## 24. Thesis recommendation

Use CSRLE to **bound claims**: the Hybrid residual learner is *capable in principle* under idealized injected structure, but real Alibaba Prophet residuals appear largely **unexploited** by the fixed GRU—consistent with residual-pattern analysis Conclusion C. Cite B0 vs C0 as controlled evidence; do **not** generalize B0 success to all cloud workloads. Pair with Ridge baseline to discuss linear vs nonlinear learnability gap.
