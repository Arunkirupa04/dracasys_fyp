# Stage 4 — Results & Discussion (Peak-Aware Global GRU)

[← Evaluation Record](03-evaluation-record.md) · [Design Lock](01-design-lock.md) · [Study Overview](README.md)

**Status:** Complete  
**Locked date:** 2026-07-17  
**Official treatment run:** `experiments/peak_aware_global_validation_vs1_2026-07-17_170939/`  
**Stage 4 lock:** `phase4_discussion.json`  
**Control reference:** `experiments/global_gru_baseline_2026-07-17_121748/`  
**Stage 3 authority:** `phase3_evaluation.json` (regenerated vs corrected baseline, Phase B)

---

## 1. Objective

Interpret the locked Stage 3 measurements for Peak-Aware Global GRU — not by retraining, redesigning methodology, or introducing new experiments. Stage 4 converts verified paired-comparison metrics into a thesis-ready discussion: what was found, what can be concluded under the evaluated configuration, plausible explanations, architecture sensitivity relative to the locked Hybrid Peak-Aware study, and limitations.

**Result authority:** `phase3_evaluation.json`, `evaluation/comparison_table.csv`, `evaluation/per_container_comparison.csv`  
**Comparison regenerated:** 2026-07-17 after Global baseline correction (Phase B); treatment model unchanged.

---

## 2. Locked Inputs (From Stage 3 — Do Not Recompute)

| Item | Value |
|------|-------|
| Cohort | 99 evaluable containers (`c_14674` skipped) |
| Horizon | Day 1 — 96 validation timesteps per container |
| Peak definition | P90 per-container, train-fitted |
| Treatment | Timestep-weighted MSE, **λ = 5**, EarlyStopping patience = 10 |
| Control | Official Global GRU baseline (`global_gru_baseline_2026-07-17_121748`, patience = 10) |
| Experimental variable | GRU training objective only |
| Verification | Stage 2 fairness 10/10 PASS; Stage 3 evaluation 10/10 PASS |

**Locked pooled results (treatment − baseline):**

| Scope | MAE Δ | RMSE Δ | Interpretation |
|-------|-------|--------|----------------|
| Overall (Tier 1) | **+0.249** | **+0.147** | Overall accuracy worsened |
| Peak subset (Tier 2) | **−0.777** | **−1.243** | Substantial peak-subset improvement |
| Non-peak subset (Tier 3) | **+0.604** | **+1.047** | Non-peak degradation |

**Per-container:** mean ΔMAE +0.249; **23 improved / 76 worsened**; peak MAE improved on **89 / 98** containers with ≥1 peak step.

---

## 3. Phase 6 Constraints (Inherited)

| Constraint | Rule |
|------------|------|
| No primary retraining | Locked VS-1 treatment and official Global baseline unchanged |
| No methodology redesign | Peak definition, λ, fairness contract, eval protocol frozen |
| Findings vs explanations | Label claims explicitly |
| Scope boundary | Conclusions apply to **this configuration on Global GRU** only |

---

## 4.1 Results Synthesis

**Status:** Complete · **Label:** *Experimental findings only*

#### Evaluation context

| Item | Value |
|------|-------|
| Control | Official Global GRU v1 (`global_gru_baseline_2026-07-17_121748`) |
| Treatment | Peak-Aware Global GRU (VS-1, λ = 5, patience = 10) |
| Cohort | 99 containers |
| Pooled peak step fraction | **25.65%** |
| Delta convention | treatment − baseline |

#### Tier 1 — Overall Day-1 metrics

| Metric | Baseline | Treatment | Δ | Relative Δ |
|--------|----------|-----------|---|------------|
| **MAE** | 1.924 | 2.174 | **+0.249** | +12.9% |
| **RMSE** | 2.611 | 2.758 | **+0.147** | +5.6% |
| **MAPE** | 116.50 | 163.77 | **+47.27** | +40.6% |

**Finding:** Overall Day-1 accuracy **worsened** under peak-aware training on Global GRU.

#### Tier 2 — Peak-subset metrics (contribution)

| Metric | Baseline | Treatment | Δ | Relative Δ |
|--------|----------|-----------|---|------------|
| **Peak MAE** | 3.426 | 2.649 | **−0.777** | −22.7% |
| **Peak RMSE** | 6.559 | 5.316 | **−1.243** | −19.0% |

**Finding:** Peak-subset error **decreased substantially**. The Tier 2 contribution objective **was met** on Global GRU.

#### Tier 3 — Non-peak-subset metrics

| Metric | Baseline | Treatment | Δ |
|--------|----------|-----------|---|
| **Non-peak MAE** | 1.406 | 2.010 | **+0.604** |
| **Non-peak RMSE** | 2.455 | 3.502 | **+1.047** |

**Finding:** Non-peak timesteps **degraded**, offsetting peak gains at the overall level.

#### Per-container paired deltas

| Statistic | ΔMAE |
|-----------|------|
| Mean | +0.249 |
| Std | 1.054 |
| Min (best for treatment) | −3.506 |
| Max (worst for treatment) | +8.206 |

| Outcome | Containers |
|---------|------------|
| Improved overall (ΔMAE < 0) | **23** |
| Worsened overall (ΔMAE > 0) | **76** |
| Improved peak MAE (≥1 peak step) | **89 / 98** |

**Machine-readable export:** `discussion/results_synthesis.json`

---

## 4.2 Answer to the Research Question

#### Tier 1 — Overall Day-1 accuracy

**Question:** Does peak-aware training change overall Day-1 accuracy vs the official Global GRU baseline?

**Answer:** **Yes — negatively.** Aggregate MAE increased by **+0.249** (+12.9%) and RMSE by **+0.147** (+5.6%). Overall Tier 1 is **not improved**.

#### Tier 2 — Peak-subset accuracy (contribution question)

**Question:** Does peak-aware training improve accuracy specifically on peak timesteps?

**Answer:** **Yes.** Pooled peak-subset MAE decreased by **−0.777** (−22.7%) and peak RMSE by **−1.243** (−19.0%). Peak-aware timestep weighting **achieved its intended peak-subset effect** on Global GRU under P90 + λ = 5.

#### Consolidated conclusion

Under the evaluated configuration — per-container **P90** peak definition, **timestep-weighted MSE with λ = 5**, and an otherwise identical Global GRU pipeline — Peak-Aware Global GRU **improves peak-subset Day-1 accuracy** but **degrades overall and non-peak accuracy**. The result is a **train/eval objective trade-off**: the model shifts error reduction toward peak timesteps at the cost of non-peak performance.

Per-container outcomes are **heterogeneous** (23 improved, 76 worsened overall), but peak MAE improved on a **large majority** of containers (89/98).

#### Scope boundary

These conclusions apply **only** to Global GRU with this peak-aware training design. They do **not** generalize to all architectures, peak definitions, or λ values.

---

## 4.3 Mechanism Discussion

**Label:** *Possible explanations* — not confirmed by Stage 3 metrics alone.

1. **Train–select–evaluate objective alignment** *(Possible explanation)* — Training minimized weighted MSE; early stopping and evaluation used unweighted metrics. Peak gains may reflect training emphasis that is not fully preserved in overall Day-1 scoring.

2. **Direct end-to-end Global GRU vs Hybrid decomposition** *(Possible explanation)* — On Global GRU, peak-aware weighting affects the **full forecast path**. On Hybrid, only the GRU **residual** path was weighted; Prophet handled trend/seasonality separately. Global architecture may expose peak weighting more directly to peak timesteps — consistent with Tier 2 success here vs Hybrid Phase 5.

3. **Peak label distribution shift** *(Possible explanation)* — Training peak weight fraction (~17%) vs evaluation peak fraction (25.65%) may affect generalization.

4. **Fixed λ = 5** *(Possible explanation)* — Locked analytically; not tuned on validation peak metrics. Other λ values untested in the primary experiment.

5. **Non-peak degradation as reallocation** *(Possible explanation)* — Model capacity may reallocate fit from non-peak to peak timesteps when peak steps receive 5× loss weight.

---

## 4.4 Stratified Discussion

**Export:** `discussion/stratified_summary.csv`

#### By validation peak fraction (quartiles)

| Peak fraction quartile | n | Mean ΔMAE (overall) | Mean Δpeak MAE | Improved (overall) |
|------------------------|---|---------------------|----------------|--------------------|
| 0–11% | 25 | +0.783 | −1.502 | 1 |
| 11–19% | 26 | +0.364 | −0.815 | 3 |
| 19–29% | 23 | +0.089 | −0.740 | 6 |
| 29–100% | 25 | **−0.255** | −0.817 | **13** |

**Findings:**
- Containers with **higher** Day-1 peak fractions show **better overall ΔMAE** (upper quartile mean −0.255).
- **Peak-subset ΔMAE is negative in every quartile** — peak improvement is consistent across peak-fraction bins.
- One container (`c_10034`) had zero peak timesteps in Day-1.

#### By workload pattern type

| Pattern | n | Mean ΔMAE (overall) | Mean Δpeak MAE | Improved | Worsened |
|---------|---|---------------------|----------------|----------|----------|
| **Spiky** | 31 | +0.547 | **−1.452** | 5 | 26 |
| Medium | 33 | +0.038 | −1.001 | 11 | 22 |
| Stable | 35 | +0.185 | −0.517 | 7 | 28 |

**Findings:**
- **Spiky** containers show the largest peak-subset gains (−1.45 mean Δpeak MAE) but also the largest overall deterioration (+0.55 mean ΔMAE).
- No pattern group shows uniform overall improvement.

---

## 4.7 Architecture Sensitivity Synthesis (Secondary)

Cross-architecture comparison using locked Peak-Aware Hybrid results (`peak_aware_2026-07-14_164518`) and this study. **Same methodology** (P90, λ = 5, timestep-weighted MSE); **different forecasting architecture**.

| Architecture | Control baseline MAE | Treatment MAE | Δ Overall MAE | Δ Peak MAE | Δ Non-peak MAE | Containers improved (overall) |
|--------------|---------------------|---------------|---------------|------------|----------------|-------------------------------|
| **Hybrid Prophet + GRU** | 1.746 | 1.741 | **−0.005** | **+0.024** | −0.015 | 45 / 99 |
| **Global GRU** | 1.924 | 2.174 | **+0.249** | **−0.777** | +0.604 | 23 / 99 |

**Findings (architecture sensitivity):**

1. **Peak-subset objective:** Achieved on **Global GRU** (−0.777 peak MAE); **not achieved** on Hybrid (+0.024 peak MAE). The shared peak-aware methodology produces **architecture-dependent peak outcomes**.

2. **Overall objective:** **Comparable** on Hybrid (−0.005 MAE); **degraded** on Global (+0.249 MAE). Global track pays a larger overall cost for peak emphasis.

3. **Trade-off pattern:** Hybrid shows near-tie overall with slight peak loss; Global shows substantial peak gain with non-peak and overall degradation. This supports the thesis framing of **one methodology, two architectures, divergent outcomes**.

4. **Interpretation (careful):** Architecture sensitivity is **descriptive synthesis** across two locked experiments — not a third controlled trial. Causal claims about “why Global responds differently” remain possible explanations (§4.3).

---

## 4.5 Limitations

1. **Single configuration** — P90 + λ = 5 on Global GRU only.
2. **λ not swept** in the primary experiment.
3. **Train/eval peak-rate divergence** (~17% train weight fraction vs 25.65% eval peak fraction).
4. **Near-idle P90 artefact** — documented in Phase 2 exploration; not corrected.
5. **MAPE instability** — near-zero CPU inflates MAPE; interpret alongside MAE/RMSE.
6. **Single trace, 99-container cohort** — limited external validity.
7. **Baseline correction (Phase B)** — comparison regenerated against corrected Global baseline (patience = 10); superseded reference preserved for history.
8. **Negative overall outcome with positive peak outcome** — valid controlled finding, not a protocol failure.

---

## 4.6 Threats to Validity

| Category | Note |
|----------|------|
| Internal | Stage 2 fairness 10/10; Stage 3 verification 10/10; control reproducibility gate PASS |
| External | Single Alibaba trace cohort; Day-1 horizon only |
| Construct | P90 operational peak proxy may differ from deployment peaks |
| Conclusion | Architecture sensitivity (§4.7) is cross-study synthesis — scope carefully in thesis |

---

## 4.8 Thesis Handoff

#### Results chapter — suggested structure

1. Experimental design recap (one controlled variable; Global GRU architecture).
2. Overall metrics (Tier 1) — Table from §4.1; emphasize +0.249 MAE.
3. Peak-subset metrics (Tier 2) — peak MAE 3.426 vs 2.649; state Tier 2 **achieved**.
4. Non-peak and per-container — §4.1 Tier 3 + 23/76 split.
5. Figures — `evaluation/plots/peak_subset_comparison.png`, error distribution, actual vs predicted.
6. Architecture sensitivity table — §4.7 (secondary, thesis §4.7).

#### Discussion chapter — suggested structure

1. Answer to research question — §4.2 (Tier 1 vs Tier 2 divergence).
2. Mechanism discussion — §4.3 (labelled possible explanations).
3. Heterogeneity — §4.4 stratified findings.
4. Architecture sensitivity — §4.7 vs Hybrid Peak-Aware.
5. Limitations — §4.5–§4.6.

#### Key sentences (copy-ready, scope-bound)

- *"Peak-Aware Global GRU improved pooled peak-subset Day-1 MAE by 0.777 (−22.7%) but worsened overall MAE by 0.249 (+12.9%) under P90 + λ = 5."*
- *"Peak-subset gains were consistent across peak-fraction quartiles; overall improvement concentrated in high peak-fraction containers."*
- *"The same peak-aware methodology produced peak-subset improvement on Global GRU but not on Hybrid Prophet + GRU — an architecture-sensitive outcome."*

#### Primary artefact index

| Need | Source |
|------|--------|
| Locked numbers | `phase3_evaluation.json`, `discussion/results_synthesis.json` |
| Full discussion | This document |
| Stage 4 lock | `phase4_discussion.json` |
| Per-container analysis | `evaluation/per_container_comparison.csv` |

---

## 5. Stage 4 Task Summary

| Task | Focus | Status |
|------|-------|--------|
| 1 | Results synthesis | **Complete** |
| 2 | Answer to research question | **Complete** |
| 3 | Mechanism discussion | **Complete** |
| 4 | Stratified discussion | **Complete** |
| 5 | Limitations and validity | **Complete** |
| 6 | Lock Stage 4 record | **Complete** |

---

## 6. Exit Criteria

- [x] §4.1 Results synthesis written (findings only)
- [x] §4.2 Research question answered (scope-bound)
- [x] §4.3 Mechanism discussion (explanations labelled)
- [x] §4.4 Stratified discussion + `stratified_summary.csv`
- [x] §4.7 Architecture sensitivity synthesis (Hybrid vs Global)
- [x] §4.5–§4.6 Limitations and validity
- [x] §4.8 Thesis handoff
- [x] `phase4_discussion.json` saved
- [x] Stage 3 evaluation artifacts unchanged (interpretation only)

---

## 7. Final Stage 4 Conclusion

Peak-Aware Global GRU (VS-1, λ = 5, P90) **achieves the peak-subset contribution objective** on the Global architecture but **fails the overall Day-1 objective**, with substantial non-peak degradation. This diverges from the locked Hybrid Peak-Aware outcome (near-tie overall, no peak gain), establishing **architecture sensitivity** as a thesis-level finding under a shared peak-aware methodology.

The Peak-Aware Global GRU study track (Stages 1–4) is **complete**.

---

*Stage 4 locked 2026-07-17 — Peak-Aware Global GRU arc complete*
