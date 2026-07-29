# CSRLE Discussion

**Status:** Stage 2 complete (2026-07-25). See [stage2_report.md](stage2_report.md) for full Q1–Q10 answers.

---

## Stage 2 — Principal findings

### Controlled positive control (B0 vs C0)

Under deterministic B0-LC injection that survives Prophet (pre-GRU VS ≈ 0.002), the fixed Hybrid GRU shows **statistically detectable but modest** residual-learning gains versus matched IID C0:

- Within-trajectory Pearson r: 0.084 vs 0.049 (bootstrap 95% CI for difference excludes zero)
- Predicted/actual std ratio: 0.073 vs 0.043 (94% of containers favor B0)
- R² remains **negative** for both (−0.33 vs −0.35)

This pattern supports a **limited capacity** to exploit injected temporal structure—not strong successful learning.

### Stochastic condition (B1 vs C1)

B1 shows a similar but weaker advantage over C1 on correlation and std ratio. Horizon-band analysis does **not** support a clear short-horizon → long-horizon decay hypothesis for B1.

### Real-data reference (A)

Condition A remains the weakest absolute residual learner (r ≈ 0.016, std ratio ≈ 0.050). B0 improves on these metrics but does not approach strong R² or high correlation.

### Ridge vs GRU

On B0, Ridge (96→96 linear) achieves **higher** residual Pearson r (0.158) than GRU (0.084) at similar MAE. This suggests injected structure is **linearly forecastable** to a greater degree than the fixed GRU captures—training/architecture may limit nonlinear exploitation.

### CPU-level translation

Hybrid Day-1 MAE improves over Prophet by only **0.006** CPU points on B0; B1/C0/C1 are flat or slightly worse. Residual-learning diagnostics remain primary; CPU gains are negligible.

---

## Interpretation boundary (mandatory)

Even where B0 exceeds C0, **do not** conclude that the Hybrid GRU is generally effective for real cloud residuals. The appropriate claim:

> Under controlled synthetic conditions with persistent forecastable nonlinear temporal structure surviving Prophet decomposition, the fixed Hybrid residual-learning pipeline demonstrated **limited** capacity to recover such structure—stronger than matched IID controls but weaker than linear Ridge and insufficient to yield positive R² or meaningful CPU improvement.

Compare explicitly with Condition A.

---

## Relationship to residual-pattern analysis

Stage 2 aligns with Conclusion C from `notebooks/residual_pattern_analysis.ipynb`: real validation residuals contain temporal structure (ACF, sign persistence), but the frozen Hybrid GRU **does not exploit it effectively**. CSRLE shows the pipeline can move slightly when structure is **injected and known**, but real Alibaba residuals still leave the learner near zero-variance corrections.

---

## Pre-GRU observations (historical)

### v1 clip gate → Amendment 1

v1 failed clip gate (~9–11% containers); v2 boundary-safe α_c resolved this. B0 retention passed in both versions (Prophet did not absorb limit-cycle via ΔP).

---

## Limitations

1. R² negative across all conditions  
2. Effect sizes modest; MAE differences near zero  
3. Ridge outperforms GRU on B0 — nonlinear learner under-delivers  
4. 8 α=0 containers in cohort (full 99 retained by design)  
5. GRU training not TF-seed locked; exact weight reproduction not expected  
6. Retention metrics (pre-GRU) are linear diagnostics; nonlinear retention not fully characterized  

---

## Thesis use

- **Support:** Fixed pipeline has non-zero residual-learning capacity under idealized B0  
- **Bound:** Real-data A remains weak; do not overclaim Hybrid residual component  
- **Contrast:** B0 vs C0 as controlled experiment; A as real reference only  
- **Future work (out of scope):** GRU architecture/training changes, not permitted in CSRLE
