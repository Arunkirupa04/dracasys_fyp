# Phase 6 — Discussion and Comparison

[← Overview](README.md) · [Phase 5 Evaluation](phase-05-evaluation.md) · [Phase 5 lock JSON](../experiments/peak_aware_2026-07-14_164518/phase5_evaluation.json) · [Phase 6 lock JSON](../experiments/peak_aware_2026-07-14_164518/phase6_discussion.json)

---

## 7. Phase 6 — Discussion and Comparison

### 7.1 Objective

Complete the Peak-Aware Hybrid research track by **interpreting the locked Phase 5 results** — not by introducing new research questions, redesigning methodology, or re-running the primary experiment.

Phase 6 converts verified measurements into a thesis-ready discussion: what was found, what can be concluded under the evaluated configuration, what may explain the outcome, and what limitations bound the claims.

**Result authority:** `experiments/peak_aware_2026-07-14_164518/phase5_evaluation.json`  
**Evaluation artefacts:** `experiments/peak_aware_2026-07-14_164518/evaluation/`  
**Control reference:** `experiments/baseline_reference_2026-07-14/`

### 7.2 Locked Inputs (From Phase 5 — Do Not Recompute)

| Item | Value |
|------|-------|
| Cohort | 99 evaluable containers (`c_14674` skipped) |
| Horizon | Day 1 — 96 validation timesteps per container |
| Peak definition | P90 per-container, train-fitted — `peak/peak_thresholds.pkl` |
| Treatment | Timestep-weighted residual MSE, **λ = 5** |
| Control | Frozen baseline Hybrid Prophet + GRU |
| Experimental variable | GRU training objective only |
| Verification | Phase 4 fairness 8/8 PASS; Phase 5 evaluation 8/8 PASS |

**Locked pooled results (treatment − baseline):**

| Scope | MAE Δ | RMSE Δ | Interpretation |
|-------|-------|--------|----------------|
| Overall (Tier 1) | **−0.005** | **−0.003** | Negligible improvement |
| Peak subset (Tier 2) | **+0.024** | **+0.018** | No peak-subset improvement |
| Non-peak subset (Tier 3) | **−0.015** | **−0.040** | Slight non-peak improvement |

**Per-container:** mean ΔMAE −0.005; **45 improved / 54 worsened**; std(ΔMAE) ≈ 0.10.

### 7.3 Phase 6 Constraints

| Constraint | Rule |
|------------|------|
| No primary retraining | Do not modify or replace the locked λ = 5 experiment |
| No methodology redesign | Do not change peak definition, fairness contract, or eval protocol |
| No baseline changes | Frozen `utils/hybrid_*.py`, `models/`, `baseline_reference/` remain read-only |
| Findings vs explanations | Label claims explicitly (see §7.4) |
| Optional work | Error attribution and λ sensitivity are **not** required for Phase 6 completion |
| Scope boundary | Conclusions apply to **this configuration only** — not all peak-aware methods |

### 7.4 Writing Standard — Findings vs Possible Explanations

All Phase 6 prose must distinguish:

| Label | Definition | Example phrasing |
|-------|------------|------------------|
| **Experimental finding** | Directly supported by locked Phase 5 metrics | “Pooled peak-subset MAE increased by 0.024 under the evaluated configuration.” |
| **Possible explanation** | Interpretation or hypothesis not independently measured in Phase 5 | “One possible explanation is that peak-period error may originate partly outside the GRU residual path.” |

**Do not** present possible explanations as proven facts.

| Avoid (overstated) | Prefer (careful) |
|--------------------|------------------|
| “Prophet dominates peak error.” | “A possible explanation is that peak-period error may not be fully addressable by reweighting GRU residual learning alone.” |
| “Peak-aware learning does not work.” | “Under P90 + λ = 5 timestep-weighted residual training, peak-subset accuracy did not improve.” |
| “λ = 5 was too low.” | “λ = 5 was locked analytically; sensitivity to other λ values was not evaluated in the primary experiment.” |

### 7.5 Research Questions (Locked From Phase 3–5)

| Tier | Question |
|------|----------|
| **Tier 1** | Does peak-aware training change overall Day-1 accuracy vs baseline? |
| **Tier 2 (contribution)** | Does peak-aware training improve accuracy specifically on peak timesteps? |

Phase 6 must answer both using **only** locked Phase 5 results in §7.8 (Answer to the Research Question).

### 7.6 Task Plan

| Task | Focus | Status |
|------|-------|--------|
| 1 | Results synthesis | **Complete** |
| 2 | Answer to the research question | **Complete** |
| 3 | Mechanism discussion (possible explanations) | **Complete** |
| 4 | Stratified discussion | **Complete** |
| 5 | Limitations and threats to validity | **Complete** |
| 6 | Lock Phase 6 record | **Complete** |

**Not in primary workflow:** Error attribution (Prophet vs GRU) — see [Appendix A](#appendix-a--optional-exploratory-error-attribution).

---

### Task 1 — Results Synthesis

**Objective:** Present locked Phase 5 outcomes in structured, thesis-ready form.

**Inputs (read-only):**
- `phase5_evaluation.json`
- `evaluation/comparison_table.csv`
- `evaluation/per_container_delta.csv`
- `evaluation/peak_subset_summary.csv`
- `evaluation/plots/`

**Deliverables:**
- §7.7 Results Synthesis in this document
- Tables: overall, peak/non-peak, per-container delta summary

**Content rules:**
- Report absolute values and deltas (treatment − baseline)
- Classify effect sizes as negligible / mixed where appropriate
- **Findings only** — no causal claims in this section

**Exit criterion:** A reader can reproduce the numeric story without opening raw CSVs.

#### Methodology

1. Read locked Phase 5 artefacts only — no re-inference or metric recomputation.
2. Compiled Tier 1 (overall), Tier 2 (peak subset), Tier 3 (non-peak subset), and per-container delta tables.
3. Classified aggregate effect sizes as negligible / mixed / no peak improvement using Phase 5 numbers only.
4. Saved machine-readable synthesis to `discussion/results_synthesis.json`.

#### Saved Artifacts

| Output | Location |
|--------|----------|
| Results synthesis (document) | §7.7 below |
| Results synthesis (JSON) | `discussion/results_synthesis.json` |
| Discussion workspace | `discussion/discussion_workspace.json` |

#### Conclusions

Task 1 exit criterion met: numeric story documented; **findings only** — no causal interpretation in this section.

#### Next Task

**Task 2 — Answer to the research question** (§7.8).

---

### Task 2 — Answer to the Research Question

**Objective:** State the research conclusion immediately following Results Synthesis (§7.8), using **only** locked Phase 5 evidence.

**Placement:** §7.8 follows §7.7 directly — this is the first interpretive subsection after numeric synthesis.

**Draft conclusion (to refine in implementation, scope-bound):**

> Under the evaluated configuration — per-container P90 peak definition, timestep-weighted GRU residual learning with λ = 5, and an otherwise identical Hybrid Prophet + GRU pipeline — the proposed Peak-Aware Hybrid model did **not** produce meaningful improvements in peak-subset Day-1 prediction accuracy. Pooled peak-subset MAE and RMSE were marginally **higher** than the frozen baseline (MAE +0.024; RMSE +0.018). Overall Day-1 performance remained **comparable** to the baseline, with negligible aggregate gains (MAE −0.005; RMSE −0.003) that are unlikely to be operationally significant. Per-container outcomes were mixed (45 containers improved, 54 worsened), indicating high heterogeneity relative to the small mean effect.
>
> These findings answer the contribution question for **this specific peak-aware training design** under the stated fairness protocol. They do **not** establish that peak-aware learning in general is ineffective, nor do they evaluate alternative peak definitions, weighting schemes, architectures, or inference-time mechanisms.

**Deliverables:**
- §7.8 Answer to the Research Question
- One paragraph + Tier 1 / Tier 2 bullet answers

**Exit criterion:** Conclusion is academically careful, scope-bound, and free of over-generalization.

---

### Task 3 — Mechanism Discussion (Possible Explanations)

**Objective:** Discuss **plausible reasons** for the locked findings without claiming unmeasured mechanisms as facts.

**Topics (all framed as possible explanations unless separately supported):**

1. **Train–select–evaluate objective alignment** — weighted training loss vs unweighted early stopping vs unweighted Day-1 evaluation
2. **Intervention scope** — only GRU residual learning was modified; Prophet fitting unchanged
3. **Peak label distribution** — train-time weighted peak fraction (~17%) vs evaluation peak fraction (~25.7%)
4. **Fixed λ = 5** — analytically locked in Phase 3; not empirically tuned on peak metrics
5. **Near-idle P90 artefact** — Phase 2 carry-forward may affect which timesteps receive high training weight

**Deliverables:**
- §7.9 Mechanism Discussion
- Each hypothesis explicitly tagged **Possible explanation**

**Exit criterion:** No mechanism stated as confirmed without Phase 5 (or appendix) evidence.

---

### Task 4 — Stratified Discussion

**Objective:** Describe heterogeneity in outcomes — avoid over-generalizing from pooled means alone.

**Inputs:**
- `evaluation/per_container_delta.csv`
- `evaluation/peak_subset_per_container_comparison.csv`
- `data/container_metadata.parquet` (workload tags from preprocessing)

**Questions (findings where supported by tables; otherwise possible patterns):**
- Which containers improved vs worsened?
- Are gains/losses associated with peak fraction or workload group?
- Does pooled near-tie conceal opposing subgroup effects?

**Deliverables:**
- §7.10 Stratified Discussion
- Optional: `discussion/stratified_summary.csv` under experiment dir

**Exit criterion:** Discussion acknowledges mixed per-container outcomes documented in Phase 5.

---

### Task 5 — Limitations and Threats to Validity

**Objective:** Thesis-grade limitations tied to Phases 1–5.

**Must include:**
1. Single primary configuration (P90, λ = 5) — conclusions not transferable to all peak-aware methods
2. λ not empirically swept in primary experiment (see [Appendix B](#appendix-b--future-work-and-supplementary-experiments))
3. Train/val peak-rate divergence under fixed thresholds
4. Near-idle P90 artefact (Phase 2)
5. MAPE instability near zero CPU
6. Single trace, 99-container cohort — external validity bounded
7. Negative peak-subset result is a **valid controlled finding**, not a failed experiment

**Deliverables:**
- §7.11 Limitations
- §7.12 Threats to validity (internal / external)

**Exit criterion:** Limitations clearly separate what was tested from what was not.

---

### Task 6 — Lock Phase 6 Record

**Objective:** Mark the Peak-Aware research track discussion complete.

**Deliverables:**
- `experiments/peak_aware_2026-07-14_164518/phase6_discussion.json`
- Final sections in this document (§7.7–§7.14)
- Update `docs/peak-aware-hybrid/README.md` — Phase 6 **Complete**
- `discussion/discussion_workspace.json` (optional tracker)
- Thesis handoff block (§7.14)

**Exit criterion:** Phases 1–6 documented end-to-end; frozen baseline untouched.

---

### 7.7 Results Synthesis

**Status:** Complete  
**Task:** 1  
**Date:** 2026-07-17  
**Label:** *Experimental findings only* — no causal interpretation in this section.

#### Evaluation context

| Item | Value |
|------|-------|
| Control | Frozen baseline Hybrid Prophet + GRU (`baseline_reference_2026-07-14`) |
| Treatment | Peak-Aware Hybrid (timestep-weighted GRU residual MSE, **λ = 5**) |
| Cohort | **99** evaluable containers (`c_14674` skipped) |
| Horizon | Day 1 — **96** validation timesteps per container |
| Peak rule | `actual_cpu_real >= P90(container_train)` |
| Pooled Day-1 steps | **9,504** (peak: **2,438**; non-peak: **7,066**) |
| Peak step fraction | **25.65%** |
| Verification | Phase 5 evaluation integrity **8/8 PASS** |

All deltas below are **peak-aware − baseline** (treatment minus control).

---

#### Tier 1 — Overall Day-1 metrics (primary)

Aggregate mean across 99 containers (real CPU %):

| Metric | Baseline | Peak-aware | Δ | Relative Δ |
|--------|----------|------------|---|------------|
| **MAE** | 1.746 | 1.741 | **−0.005** | −0.29% |
| **RMSE** | 2.388 | 2.384 | **−0.003** | −0.14% |
| **MAPE** | 111.94 | 110.17 | **−1.77** | −1.58% |

**Finding:** Overall Day-1 accuracy changed by less than one third of one percent in MAE. Aggregate performance remained **comparable** to the baseline.

Per-container dispersion (MAE std across cohort):

| Model | MAE mean | MAE std | MAE min | MAE max |
|-------|----------|---------|---------|---------|
| Baseline | 1.746 | 2.487 | 0.003 | 17.676 |
| Peak-aware | 1.741 | 2.468 | 0.003 | 17.753 |

**Finding:** Cohort-level spread is similar; the small mean shift does not indicate a uniform improvement across containers.

---

#### Tier 2 — Peak-subset metrics (contribution)

Pooled across all peak-labelled Day-1 timesteps (2,438 steps):

| Metric | Baseline | Peak-aware | Δ | Relative Δ |
|--------|----------|------------|---|------------|
| **Peak MAE** | 2.501 | 2.526 | **+0.024** | +0.98% |
| **Peak RMSE** | 5.186 | 5.204 | **+0.018** | +0.34% |

**Finding:** Peak-subset error was **marginally higher** under peak-aware training. The peak-subset objective was **not met** in this evaluation.

---

#### Tier 3 — Non-peak-subset metrics

Pooled across non-peak Day-1 timesteps (7,066 steps):

| Metric | Baseline | Peak-aware | Δ |
|--------|----------|------------|---|
| **Non-peak MAE** | 1.485 | 1.470 | **−0.015** |
| **Non-peak RMSE** | 3.491 | 3.451 | **−0.040** |

**Finding:** Non-peak timesteps showed a **slight** reduction in error relative to baseline. The magnitude of non-peak RMSE improvement (−0.040) exceeds the peak MAE deterioration (+0.024) in absolute terms on their respective step pools.

---

#### Per-container paired deltas

From `evaluation/per_container_delta.csv` (99 paired containers):

| Statistic | ΔMAE | ΔRMSE |
|-----------|------|-------|
| Mean | −0.005 | −0.003 |
| Std | 0.100 | 0.107 |
| Min (best for peak-aware) | −0.867 | −0.939 |
| Max (worst for peak-aware) | +0.103 | +0.124 |

| Outcome | Containers |
|---------|------------|
| Improved (ΔMAE < 0) | **45** |
| Worsened (ΔMAE > 0) | **54** |
| Unchanged | **0** |

**Finding:** Per-container outcomes are **mixed**. The mean ΔMAE (−0.005) is small relative to the per-container standard deviation (0.100), indicating **heterogeneous** effects rather than a consistent cohort-wide shift.

---

#### Summary of experimental findings

| Question tier | Metric focus | Direction vs baseline | Magnitude |
|---------------|--------------|----------------------|-----------|
| Tier 1 — Overall | MAE / RMSE / MAPE | Slightly lower | Negligible (<0.3% MAE) |
| Tier 2 — Peak subset | Peak MAE / RMSE | Slightly higher | Small (+0.024 MAE) |
| Tier 3 — Non-peak subset | Non-peak MAE / RMSE | Slightly lower | Small (−0.015 / −0.040) |
| Per-container | ΔMAE | Mixed | 45 improved / 54 worsened |

**Source artefacts:** `phase5_evaluation.json`, `evaluation/comparison_table.csv`, `evaluation/peak_subset_summary.csv`, `evaluation/per_container_delta.csv`, `evaluation/plots/`.

**Machine-readable export:** `discussion/results_synthesis.json`

---

### 7.8 Answer to the Research Question

**Status:** Complete  
**Task:** 2  
**Date:** 2026-07-17  
**Evidence base:** §7.7 experimental findings only (`phase5_evaluation.json`)

#### Tier 1 — Overall Day-1 accuracy

**Question:** Does peak-aware training change overall Day-1 accuracy vs the frozen baseline?

**Answer:** Under the evaluated configuration, overall Day-1 accuracy remained **comparable** to the baseline. Aggregate MAE decreased by 0.005 (−0.29%) and RMSE by 0.003 (−0.14%). These changes are **negligible** relative to baseline MAE (1.746) and cohort dispersion (MAE std ≈ 2.49). They are unlikely to represent an operationally meaningful overall improvement.

#### Tier 2 — Peak-subset accuracy (contribution question)

**Question:** Does peak-aware training improve accuracy specifically on peak timesteps?

**Answer:** **No.** Pooled peak-subset MAE increased by **+0.024** (+0.98%) and peak RMSE by **+0.018** (+0.34%) relative to the baseline. The proposed method did **not** produce meaningful improvements in peak-subset Day-1 prediction accuracy under P90 labelling and λ = 5.

#### Consolidated conclusion

Under the evaluated configuration — per-container **P90** peak definition, **timestep-weighted GRU residual learning with λ = 5**, and an otherwise identical Hybrid Prophet + GRU pipeline — the Peak-Aware Hybrid model **did not** achieve the intended peak-subset improvement. Overall performance remained **statistically and practically comparable** to the frozen baseline, with a slight non-peak improvement and a slight peak-subset deterioration that largely offset one another at the cohort level.

Per-container outcomes were **mixed** (45 improved, 54 worsened), so the small positive mean overall shift must not be interpreted as uniform model superiority.

#### Scope boundary (required)

These conclusions apply **only** to the specific peak-aware training design evaluated in this study. They do **not** establish that peak-aware learning in general is ineffective, nor do they evaluate alternative peak definitions, weighting schemes (including other λ values), architecture changes, Prophet-side modifications, or inference-time peak mechanisms. The result is a **valid controlled research finding** for this configuration, not a universal verdict on peak-aware forecasting.

---

### 7.9 Mechanism Discussion

**Status:** Complete  
**Task:** 3  
**Date:** 2026-07-17  
**Label:** *Possible explanations* — not confirmed by Phase 5 primary metrics alone.

The following points **may** help interpret §7.7–§7.8. Each is a **hypothesis**, not a proven mechanism.

#### 1. Train–select–evaluate objective alignment *(Possible explanation)*

Training minimized **weighted** residual MSE (λ = 5 on peak-labelled target timesteps in training sequences). Model selection (early stopping) monitored **unweighted** sequence validation loss. Final evaluation used **unweighted** Day-1 MAE/RMSE on the temporal holdout. If peak-subset accuracy was the target, the training and selection objectives were not fully aligned with the reported peak-subset metric.

#### 2. Narrow intervention scope *(Possible explanation)*

Only the **GRU residual training objective** was modified. Prophet fitting, architecture, data, inference, and forecast combination remained identical. Peak-period error in the final forecast depends on **both** Prophet and GRU components. Reweighting GRU residual learning alone may not reduce peak CPU error if a substantial share of peak-period deviation is already present in the Prophet component or is not captured by the residual pathway.

#### 3. Peak label distribution shift *(Possible explanation)*

During training, approximately **17%** of weighted target timesteps were peak-labelled (`peak_weight_fraction ≈ 0.169` in training metadata). At evaluation, **25.65%** of Day-1 timesteps were peak-labelled under train-fitted P90 thresholds. The model may have been emphasized on a different peak frequency profile than the one used for peak-subset scoring.

#### 4. Fixed λ = 5 without empirical peak-metric tuning *(Possible explanation)*

λ = 5 was locked analytically in Phase 3 to balance peak vs non-peak loss contribution. It was **not** tuned on temporal validation peak metrics (by design, to protect evaluation integrity). Other λ values might behave differently; this was **not tested** in the primary experiment and must not be inferred from Phase 5 results.

#### 5. Near-idle P90 artefact *(Possible explanation)*

Phase 2 documented that a subset of near-idle containers can produce threshold artefacts under P90. Such containers may receive uniformly high training weights without corresponding operational peak behaviour at evaluation. This may dilute or distort the intended peak-aware signal. This artefact was **not removed** in the primary experiment.

#### Summary

Phase 5 **establishes what was observed** (§7.7) and **answers the research question for this configuration** (§7.8). The mechanisms above are **interpretive context** for thesis discussion and future work — not additional primary evidence.

---

### 7.10 Stratified Discussion

**Status:** Complete  
**Task:** 4  
**Date:** 2026-07-17  
**Inputs:** `per_container_delta.csv`, `peak_subset_per_container_comparison.csv`, `container_metadata.parquet`  
**Export:** `discussion/stratified_summary.csv`

#### Per-container heterogeneity *(Experimental finding)*

Pooled means near zero conceal opposing container-level outcomes:

| Outcome | Containers |
|---------|------------|
| Overall ΔMAE improved (peak-aware better) | **45** |
| Overall ΔMAE worsened | **54** |

The mean ΔMAE (−0.005) is an order of magnitude smaller than the per-container standard deviation (0.100). **Finding:** Cohort-level aggregates understate heterogeneity.

#### By validation peak fraction (quartiles) *(Experimental finding)*

Containers grouped by Day-1 peak timestep fraction (`peak_fraction_treatment`):

| Peak fraction quartile | n | Mean ΔMAE (overall) | Mean Δpeak MAE | Improved (overall) |
|------------------------|---|---------------------|----------------|--------------------|
| 0–11% | 25 | +0.007 | +0.023 | 13 |
| 11–19% | 26 | −0.002 | +0.046 | 11 |
| 19–29% | 23 | −0.011 | +0.025 | 12 |
| 29–100% | 25 | −0.015 | +0.027 | 9 |

**Findings:**
- Containers with **higher** Day-1 peak fractions tended toward **better overall ΔMAE** (more negative mean delta in upper quartiles).
- **Peak-subset ΔMAE remained positive in every quartile** (peak-aware peak error not lower in any bin).
- One container (`c_10034`) had **zero** peak timesteps in Day-1; peak-subset metrics are not applicable for that container.

Among 98 containers with ≥1 peak step: **24** improved peak MAE, **74** worsened (mean Δpeak MAE ≈ +0.030).

#### By workload pattern type *(Experimental finding)*

From `container_metadata.parquet` (`pattern_type`: stable / medium / spiky):

| Pattern | n | Mean ΔMAE (overall) | Improved | Worsened |
|---------|---|---------------------|----------|----------|
| **Spiky** | 31 | **−0.038** | 17 | 14 |
| Stable | 35 | +0.004 | 14 | 21 |
| Medium | 33 | +0.016 | 14 | 19 |

**Findings:**
- **Spiky** containers showed the largest mean overall improvement under peak-aware training.
- **Medium** containers showed the largest mean overall deterioration.
- No pattern group showed uniform improvement; even spiky containers were **14/31 worsened** on overall MAE.

#### Interpretation (careful)

Stratification shows the pooled near-tie is a **balance of subgroup effects**, not uniform behaviour. It does **not** overturn §7.8: peak-subset accuracy did not improve at the pooled level, and peak ΔMAE was not negative in any peak-fraction quartile. Subgroup patterns are **descriptive findings** for discussion; causal claims by workload type are **not** supported without further study.

---

### 7.11 Limitations

**Status:** Complete  
**Task:** 5  
**Date:** 2026-07-17

1. **Single evaluated configuration** — Conclusions apply to P90 + λ = 5 timestep-weighted GRU training only, not to peak-aware learning in general.

2. **λ not empirically swept in the primary experiment** — λ = 5 was locked analytically in Phase 3. Sensitivity to λ ∈ {3, 5, 10} was **not** part of the primary study (see Appendix B).

3. **Train/validation peak-rate divergence** — Training peak weight fraction (~17%) differs from evaluation peak fraction (~25.7%) under fixed train-fitted thresholds.

4. **Near-idle P90 artefact** — Phase 2 documented threshold artefacts for near-idle containers; these were not corrected in the primary pipeline.

5. **MAPE instability** — Near-zero CPU values inflate MAPE despite epsilon flooring; MAPE should be interpreted cautiously alongside MAE/RMSE.

6. **Single dataset and cohort** — Alibaba Cluster Trace, 99 evaluable containers; external validity to other clouds, workloads, or horizons is limited.

7. **One experimental variable only** — Prophet, architecture, and inference were frozen; negative peak-subset result does not test peak-aware changes in other components.

8. **Negative peak-subset outcome is a valid result** — The controlled comparison succeeded in producing an auditable answer; a non-improvement on peaks is a research finding, not a protocol failure.

9. **Stratified patterns are exploratory at subgroup level** — Pattern-type bins (n ≈ 31–35) are descriptive; they do not replace the primary pooled comparison.

---

### 7.12 Threats to Validity

**Status:** Complete  
**Task:** 5  
**Date:** 2026-07-17

#### Internal validity

| Threat | Mitigation / residual risk |
|--------|---------------------------|
| Unfair baseline comparison | Phase 4 fairness 8/8 PASS; identical data, architecture, inference |
| Evaluation leakage | Peak thresholds train-fitted only; temporal val holdout unchanged |
| Metric recomputation error | Phase 5 verification 8/8 PASS; metrics reproducible from caches |
| Objective mismatch | **Residual risk:** weighted train vs unweighted early stop vs unweighted eval may limit peak-metric gains even if weighting has some effect |

#### External validity

| Threat | Note |
|--------|------|
| Single trace / time period | Findings may not transfer to other deployment contexts |
| Selected 100-container cohort | One container skipped; not a random sample of all cluster containers |
| Day-1 horizon only | 96-step holdout; longer horizons not evaluated in this arc |
| CPU utilization only | Memory / multi-resource forecasting out of scope |

#### Construct validity

| Threat | Note |
|--------|------|
| P90 peak definition | Operational “peak” may differ from statistical P90 on train CPU |
| Peak labels on actuals at eval | Correct for scoring; training labels derived from training-period CPU — see §7.9 |

#### Conclusion validity

Primary conclusions (§7.8) are **bounded** to the evaluated design. Possible explanations (§7.9) and subgroup tables (§7.10) must not be overstated as confirmation of why the model behaved as observed.

---

### 7.13 Thesis Narrative — Actual Contribution (Locked Focus)

Phase 6 discussion and thesis prose should emphasize **what was actually delivered**:

1. A **Peak-Aware Hybrid Prophet + GRU** model was **designed** (Phase 3) with one controlled experimental variable: timestep-weighted GRU residual training.
2. The design was **implemented** (Phase 4) in parallel modules without modifying the frozen baseline.
3. **Only the GRU training objective** was changed; Prophet, data, architecture, inference, and evaluation protocol remained identical.
4. The implementation **passed all fairness verification** (Phase 4: 8/8) and evaluation integrity checks (Phase 5: 8/8).
5. Under the evaluated configuration (P90, λ = 5), the method produced **negligible overall improvements** and **did not improve peak-subset accuracy**.
6. This is a **valid controlled research finding** — reported honestly with limitations and possible explanations, not as a universal verdict on peak-aware learning.

**Do not** frame Phase 6 as “fixing” the model or “proving” why it failed. Frame it as **completing the research** with an auditable negative/refinement result.

---

### 7.14 Thesis Handoff

**Status:** Complete  
**Task:** 6  
**Date:** 2026-07-17

Use the following blocks when drafting **Results** and **Discussion** chapters.

#### Results chapter — suggested structure

1. **Experimental design recap** — One controlled variable (GRU loss weighting); frozen baseline; 99 containers; Day-1 96-step holdout; P90 peaks.
2. **Overall metrics (Tier 1)** — Table from §7.7; emphasize comparable performance (MAE 1.746 vs 1.741).
3. **Peak-subset metrics (Tier 2)** — Peak MAE 2.501 vs 2.526; state peak objective not met.
4. **Non-peak and per-container results** — §7.7 Tier 3 + 45/54 split.
5. **Figures** — `evaluation/plots/peak_subset_comparison.png`, error distribution, sample actual-vs-predicted.
6. **Fairness** — Phase 4 + Phase 5 verification PASS (brief footnote).

#### Discussion chapter — suggested structure

1. **Answer to research question** — §7.8 consolidated conclusion (scope-bound).
2. **Interpretation** — §7.9 possible explanations, clearly labelled.
3. **Heterogeneity** — §7.10 stratified findings (spiky vs medium; peak fraction quartiles).
4. **Limitations** — §7.11.
5. **Threats to validity** — §7.12 (condensed).
6. **Contribution statement** — §7.13 six-point narrative.
7. **Future work** — Appendix B only (λ sweep, error attribution as exploratory).

#### Key sentences (copy-ready, scope-bound)

- *"The Peak-Aware Hybrid model was implemented under a fair controlled protocol with only the GRU training objective modified."*
- *"Overall Day-1 forecasting accuracy remained comparable to the frozen baseline (ΔMAE −0.005)."*
- *"Peak-subset accuracy did not improve under P90 + λ = 5; pooled peak MAE was marginally higher (+0.024)."*
- *"This finding applies to the evaluated configuration and does not generalize to all peak-aware learning approaches."*

#### Primary artefact index

| Chapter need | Source |
|--------------|--------|
| Locked numbers | `phase5_evaluation.json`, `discussion/results_synthesis.json` |
| Full discussion | `docs/peak-aware-hybrid/phase-06-discussion.md` |
| Phase 6 lock | `phase6_discussion.json` |
| Manual inspection | `notebooks/peak_aware_container_comparison.ipynb` |

---

## Appendix A — Optional Exploratory: Error Attribution

**Status:** Not required for Phase 6 completion.  
**Label in thesis:** *Exploratory / supplementary analysis* — not primary evidence.

### Why this is outside the primary workflow

The primary research question was answered by the **controlled paired comparison** in Phase 5 (overall + peak-subset metrics). Error attribution investigates a **different question**: where remaining error originates (e.g., Prophet vs GRU residual path). That is useful context but **not** part of the main contribution claim.

### If implemented later

| Item | Guidance |
|------|----------|
| Question | “At peak timesteps, how much error remains after Prophet alone vs after the full hybrid?” |
| Inputs | Inference caches or `notebooks/peak_aware_container_comparison.ipynb` |
| Output location | `experiments/.../discussion/supplementary/error_attribution/` |
| Writing rule | Results labelled **exploratory**; cannot upgrade §7.8 conclusion |
| Example phrasing | “Exploratory decomposition suggests…” — not “Prophet dominates peak error.” |

**Do not** block Phase 6 lock on completion of this appendix.

---

## Appendix B — Future Work and Supplementary Experiments

**Status:** Listed for completeness only. **Not** part of the primary Phase 6 workflow or locked conclusions.

| Item | Note |
|------|------|
| **λ sensitivity (λ ∈ {3, 5, 10})** | Would require **new training runs** in separate experiment directories. Does not modify the locked λ = 5 primary experiment or its conclusions. |
| **Peak-aligned early stopping** | Exploratory design variant; not the locked fairness contract. |
| **Global GRU comparison** | Requires fixing train/val leakage first (`docs/methodology-gap-analysis.md`). Out of peak-aware primary arc unless separately scoped. |
| **Alternative peak definitions** | New research question; not a reinterpretation of Phase 5. |

---

### 7.15 Exit Criteria (Phase 6 Complete)

- [x] §7.7 Results synthesis written (findings only)
- [x] §7.8 Research question answered (scope-bound, academic tone)
- [x] §7.9 Mechanism discussion written (explanations labelled)
- [x] §7.10 Stratified discussion written
- [x] §7.11–§7.12 Limitations and threats to validity written
- [x] §7.13–§7.14 Thesis narrative and handoff finalized
- [x] `phase6_discussion.json` saved
- [x] README updated — Phase 6 complete
- [x] Frozen baseline and Phase 5 results unchanged
- [x] Appendix A **not** required
- [x] Appendix B items **not** presented as primary evidence

### 7.16 Estimated Effort

| Task | Effort |
|------|--------|
| 1 — Results synthesis | 1–2 hours |
| 2 — Research question answer | 1 hour |
| 3 — Mechanism discussion | 2–3 hours |
| 4 — Stratified discussion | 1–2 hours |
| 5 — Limitations | 1–2 hours |
| 6 — Lock record | ~1 hour |

**Total (primary workflow):** ~1–1.5 days of writing and light analysis on existing artefacts.

### 7.17 Handoff From Phase 5

Phase 5 delivered measurement only. Phase 6 delivers interpretation only.

**Start here:**
- `experiments/peak_aware_2026-07-14_164518/phase5_evaluation.json`
- `evaluation/comparison_table.csv`, `per_container_delta.csv`, `peak_subset_summary.csv`
- `notebooks/peak_aware_container_comparison.ipynb` (manual inspection)

**Recommended task order:**

```
Task 1 → Task 2 → Task 3 → Task 4 → Task 5 → Task 6
```

---

## 7.18 Phase 6 — Final Summary (Locked)

**Status:** **Complete**  
**Lock record:** `experiments/peak_aware_2026-07-14_164518/phase6_discussion.json`  
**Locked:** 2026-07-17

The Peak-Aware Hybrid research track (Phases 1–6) is **complete**. Phase 6 interpreted locked Phase 5 measurements without modifying the primary experiment, baseline code, or evaluation results.

| Task | Deliverable | Status |
|------|-------------|--------|
| 1 | §7.7 Results synthesis + `results_synthesis.json` | Complete |
| 2 | §7.8 Answer to research question | Complete |
| 3 | §7.9 Mechanism discussion (possible explanations) | Complete |
| 4 | §7.10 Stratified discussion + `stratified_summary.csv` | Complete |
| 5 | §7.11–§7.12 Limitations and validity | Complete |
| 6 | `phase6_discussion.json`, §7.14 thesis handoff, README | Complete |

**Research outcome (this configuration):** Peak-aware GRU training (P90, λ = 5) did **not** improve peak-subset Day-1 accuracy; overall performance remained **comparable** to baseline. Valid controlled finding — see §7.8.

**Optional later work:** Appendix A (error attribution), Appendix B (λ sensitivity, Global GRU) — **not** required for this lock.

---

*Phase 6 locked: 2026-07-17 — Peak-Aware Hybrid arc complete (Phases 1–6)*
