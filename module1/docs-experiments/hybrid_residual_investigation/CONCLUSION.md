# Hybrid Residual Investigation — Consolidated Conclusion

**Scope:** Stage 0 (Residual Representation Diagnostics) + RRE v1.0 (Residual Scaling Experiment)  
**Cohort:** 99 evaluable containers, frozen Hybrid research protocol  
**Artifacts:**
- Stage 0: `experiments/rre_diagnostics_2026-07-27_120759/`
- RRE v1.0: `experiments/rre_2026-07-27_123441/`

---

## Thesis statement

> After systematic diagnostic and controlled experimental evaluation, the Prophet complement residual is best modelled in **level form with global z-score normalisation**. Alternative temporal encodings (velocity) and robust scalings (median/MAD) do not improve Hybrid GRU forecasting. Residual preprocessing is not the bottleneck for Hybrid performance under the frozen protocol.

---

## Research arc (two stages, one narrative)

| Stage | Type | Question | Outcome |
|-------|------|----------|---------|
| **Stage 0** | Read-only statistics | Which representation preserves temporal structure? | **R0** strongest; **R1 rejected**; **R3** same ACF as R0, better numerical properties |
| **RRE v1.0** | GRU training (R0 vs R3) | Does R3's conditioning improve optimisation/accuracy? | **Null result** — retain R0 |

Stage 0 chose the challenger **evidence-driven** (velocity was not assumed optimal). RRE v1.0 tested the only remaining plausible hypothesis: that identical temporal information, differently scaled, would ease GRU optimisation.

---

## Stage 0 conclusions

### Velocity (R1) is rejected

First-order differencing **removes** temporal dependence rather than exposing it:

| Metric | R0 | R1 |
|--------|----|----|
| mean \|ACF\| (lags 1–96) | 0.0348 | **0.0116** |
| Integrated autocorrelation τ (median) | 0.54 | **≈ 0.007** |
| Sparsity (\|x\| < 0.05) | 10.5% | **27.3%** |

**Conclusion:** Velocity must not be used as the Hybrid residual target.

### R0, R2, R3 preserve the same temporal structure

Affine scaling does not change autocorrelation. R0, R2, and R3 share identical cohort mean ACF/PACF (lag-1 ACF ≈ 0.36). They differ only in **distribution shape**:

| ID | Skew | Sparsity (<0.05) | Role |
|----|------|------------------|------|
| R0 | 1.43 | 10.5% | Frozen Hybrid baseline |
| R2 | 3.88 | 6.6% | No clear advantage; worse kurtosis |
| R3 | 1.43 | **3.3%** | Same temporal info; lower sparsity |

**Conclusion:** Representation selection is not about discovering new temporal information — it is about numerical conditioning of the **same** signal.

---

## RRE v1.0 conclusions

### Primary finding: null result

Robust median/MAD scaling (R3) does **not** improve Hybrid forecasting:

| Metric | R0 | R3 | Δ (R3−R0) | Wilcoxon p |
|--------|----|----|-----------|------------|
| Day-1 MAE (%) | **1.740** | 1.745 | +0.0052 | 0.975 |
| Day-1 RMSE (%) | **2.380** | 2.385 | +0.0050 | 0.161 |
| Residual Pearson r | 0.047 | 0.050 | +0.003 | 0.709 |

- **Not statistically significant** for primary CPU metrics
- **Not practically meaningful** (Δ ≪ 0.05% threshold)
- **Do not replace** global z-score with robust scaling

### Optimisation behaviour

| Metric | R0 | R3 |
|--------|----|----|
| Best epoch | 5 | 4 |
| Epochs until early stop | 15 | 14 |

R3 converged one epoch earlier, but this did **not** translate into better forecasts. Raw MSE loss values are **not comparable** across arms (R3 scaled residuals have ~7× higher variance by construction).

### Eight research questions — final answers

1. **Accuracy improved?** No  
2. **Optimisation improved?** No (marginal epoch gain only)  
3. **Convergence speed?** Marginally (best epoch 4 vs 5)  
4. **Generalisation?** No  
5. **Statistically significant?** No (primary metrics)  
6. **Practically meaningful?** No  
7. **Replace baseline scaling?** **No — retain R0 global z-score**  
8. **Optimisation vs temporal information?** Stage 0 showed identical temporal structure; RRE confirms scaling alone does not unlock better learning  

---

## What this means in context of all Hybrid experiments

The following interventions did **not** yield meaningful Hybrid gains under the frozen protocol:

| Experiment | Variable tested | Result |
|------------|-----------------|--------|
| Peak-aware loss | Training loss weighting | No significant gain |
| HCERL | Context features | No significant gain |
| GGTCE / TMA | Input window length | 96 steps sufficient |
| Stage 0 + RRE | Residual representation & scaling | **R1 rejected; R3 null** |

Hybrid improves over Prophet (~78% variance explained), but **further gains are not coming from how the residual is encoded, scaled, contextualised, or windowed** under the current GRU architecture and MSE loss.

The limitation likely lies elsewhere:

- Prophet ceiling on this cohort
- GRU architecture / capacity
- MSE on a near-zero, heavy-tailed residual
- Intrinsic difficulty of 96-step residual forecasting

---

## Critical distinction (for thesis examiners)

| Question | Answered by | Answer |
|----------|-------------|--------|
| *Does a better temporal representation exist?* | Stage 0 | Velocity **no**; level residual **yes** |
| *Does better scaling of the same signal help the GRU?* | RRE v1.0 | **No** |
| *Should we change the Hybrid residual target?* | Both stages | **No — keep level + global z-score** |

This is **not** a failed research programme. A rigorous null result closes the residual-preprocessing hypothesis and strengthens the justification for the frozen Hybrid baseline used in production.

---

## Recommendations

### For the Hybrid model (research and production)

- **Retain R0:** level Prophet residual with global train z-score
- **Do not adopt** velocity differencing or median/MAD scaling
- **Do not pursue** further representation/scaling ablations unless architecture or loss also changes

### For future research

Residual preprocessing is **closed** as a thread. Future work should consider:

- Prophet model improvements
- Alternative losses (e.g. Huber) — potentially interacting with scaling
- Architecture changes (depth, attention, multi-scale)
- Production-scale validation (435-container cohort)

---

## Where to find supporting evidence

| Document | Content |
|----------|---------|
| `docs/residual_representation_diagnostics/FINDINGS.md` | Stage 0 statistical evidence |
| `docs/residual_scaling_experiment/results.md` | RRE v1.0 numbers and tests |
| `docs/residual_scaling_experiment/discussion.md` | Interpretation and null-result framing |
| `experiments/rre_diagnostics_2026-07-27_120759/reports/final_report.md` | Stage 0 recommendation |
| `experiments/rre_2026-07-27_123441/reports/final_report.json` | RRE answers (machine-readable) |
| `notebooks/residual_representation_diagnostics.ipynb` | Stage 0 figures |
| `notebooks/residual_scaling_experiment.ipynb` | RRE figures and case studies |

---

## Suggested thesis paragraph (copy-ready)

Prior experiments established that Hybrid Prophet+GRU consistently outperforms Prophet alone, but modifications to loss weighting (peak-aware), input context (HCERL), and window length (GGTCE) did not produce significant further gains. We hypothesised that the remaining limitation might lie in how the Prophet complement residual is represented for GRU learning. A two-stage investigation was therefore conducted. Stage 0 (Residual Representation Diagnostics) compared causal representations using read-only statistics and rejected velocity differencing, which substantially reduced temporal autocorrelation, while confirming that robust median/MAD scaling preserves identical autocorrelation structure to the baseline level residual. RRE v1.0 then trained two Hybrid models differing only in scaling method (global z-score vs median/MAD) under an otherwise frozen protocol. Robust scaling did not improve Day-1 MAE (1.740% vs 1.745%, Wilcoxon p = 0.975) or produce practically meaningful gains. We conclude that the Prophet complement residual is appropriately modelled in level form with global z-score normalisation, and that residual preprocessing is not the primary bottleneck for Hybrid performance under the current architecture.
