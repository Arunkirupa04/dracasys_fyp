# Final Four-Model Comparison — Synthesis Summary

**Generated:** 2026-07-23T07:08:06Z  
**Interpretation revised:** 2026-07-23 (academic review — metrics unchanged)  
**Type:** Read-only synthesis from locked evaluation outputs (no retraining, no inference regeneration).

---

## Master comparison table

| Model | Overall MAE | Overall RMSE | Peak MAE | Peak RMSE | Non-Peak MAE | Non-Peak RMSE |
|-------|-------------|--------------|----------|-----------|--------------|---------------|
| Hybrid Prophet + GRU (Baseline) | 1.746 | 2.388 | **2.501** | **5.186** | 1.485 | 3.491 |
| Peak-Aware Hybrid Prophet + GRU | 1.741 | 2.384 | 2.526 | 5.204 | 1.470 | 3.451 |
| Global GRU (Official Baseline) | 1.924 | 2.611 | 3.426 | 6.559 | **1.406** | **2.455** |
| Peak-Aware Global GRU | 2.174 | 2.758 | 2.649 | 5.316 | 2.010 | 3.502 |

*Values rounded for presentation; authoritative locked values remain in `master_metrics_table.csv` and source experiment artifacts.*

---

## Rankings (rank 1 = best / lowest error)

Rankings describe numerical ordering only. **Small absolute gaps (especially on the Hybrid track) should not be interpreted as meaningful performance differences without further statistical testing.**

- **overall_mae:** 1. Peak-Aware Hybrid (1.741), 2. Hybrid Baseline (1.746), 3. Global Baseline (1.924), 4. Peak-Aware Global (2.174)
- **overall_rmse:** 1. Peak-Aware Hybrid (2.384), 2. Hybrid Baseline (2.388), 3. Global Baseline (2.611), 4. Peak-Aware Global (2.758)
- **peak_mae:** 1. **Hybrid Baseline (2.501)**, 2. Peak-Aware Hybrid (2.526), 3. Peak-Aware Global (2.649), 4. Global Baseline (3.426)
- **peak_rmse:** 1. **Hybrid Baseline (5.186)**, 2. Peak-Aware Hybrid (5.204), 3. Peak-Aware Global (5.316), 4. Global Baseline (6.559)
- **non_peak_mae:** 1. Global Baseline (1.406), 2. Peak-Aware Hybrid (1.470), 3. Hybrid Baseline (1.485), 4. Peak-Aware Global (2.010)
- **non_peak_rmse:** 1. Global Baseline (2.455), 2. Peak-Aware Hybrid (3.451), 3. Hybrid Baseline (3.491), 4. Peak-Aware Global (3.502)

---

## Architecture sensitivity summary (thesis-ready)

Peak-Aware methodology was identical across tracks: **P90 train-fitted thresholds**, **λ = 5**, **timestep-weighted MSE** on forecast-horizon targets, unweighted validation early stopping, and the same 99-container Day-1 evaluation protocol. Only the underlying forecasting architecture differed.

| Architecture | Baseline | Peak-Aware | Overall effect (ΔMAE) | Peak effect (Δpeak MAE) | Non-peak effect (Δnon-peak MAE) | Overall conclusion |
|--------------|----------|------------|------------------------|-------------------------|----------------------------------|--------------------|
| **Hybrid** (Prophet + GRU) | Hybrid Baseline (1.746) | Peak-Aware Hybrid (1.741) | **Negligible** (−0.005; −0.3%) | **Negligible / slightly worse** (+0.024; +1.0%) | **Negligible** (−0.015; −1.0%) | **No meaningful benefit** from Peak-Aware training under this configuration |
| **Global** (direct GRU) | Global Baseline (1.924) | Peak-Aware Global (2.174) | **Worse** (+0.249; +13.0%) | **Much better** (−0.777; −22.7%) | **Worse** (+0.604; +42.9%) | **Architecture-dependent trade-off** — peak gains at substantial non-peak and overall cost |

*Deltas are treatment − baseline within each architecture, from locked paired comparisons.*

---

## Academic interpretation

### 1. Best overall model

**Finding:** The **Hybrid Prophet + GRU architecture** provides the best overall Day-1 forecasting performance in this study. Both Hybrid variants (MAE 1.746 and 1.741) substantially outperform both Global variants (MAE 1.924 and 2.174).

**Peak-Aware Hybrid vs Hybrid Baseline:** The aggregate difference is **0.005 MAE** (0.003 RMSE) — approximately **0.3%** relative to the Hybrid baseline. This is **not a meaningful operational improvement** and is well within the scale of per-container heterogeneity reported in the locked Peak-Aware Hybrid evaluation (45 containers improved vs 54 worsened on overall MAE).

**Recommended thesis wording:** *"The Hybrid Prophet + GRU architecture achieves the lowest overall Day-1 error among all evaluated models. Peak-Aware Hybrid performs comparably to the Hybrid baseline, with no meaningful overall improvement (+0.005 MAE)."*

Ranking Peak-Aware Hybrid first on overall MAE is numerically correct but **should not be overstated** as a distinct practical advantage over Hybrid Baseline.

---

### 2. Best peak prediction — verified conclusion

**Verified:** **Hybrid Baseline** has the **lowest peak-subset MAE (2.501)** and **lowest peak-subset RMSE (5.186)** across all four final models.

| Model | Peak MAE | Rank |
|-------|----------|------|
| Hybrid Baseline | **2.501** | 1 |
| Peak-Aware Hybrid | 2.526 | 2 |
| Peak-Aware Global | 2.649 | 3 |
| Global Baseline | 3.426 | 4 |

**Why this matters academically:**

1. **Peak-Aware training did not improve peak-subset accuracy on the Hybrid track** — the intended Tier-2 objective was not achieved (peak MAE +0.024 vs Hybrid Baseline). The locked Hybrid Phase 6 discussion reached the same conclusion for the evaluated configuration.

2. **Peak-Aware Global improves peak accuracy relative to Global Baseline only** (3.426 → 2.649) but **does not surpass either Hybrid model** on pooled peak MAE. Cross-architecture peak performance is dominated by the Hybrid pipeline, not by Peak-Aware Global.

3. **The best peak forecaster is therefore the unmodified Hybrid Baseline**, not a Peak-Aware variant. This contradicts the original motivation that timestep-weighted training would preferentially improve peak timesteps, at least under P90 + λ = 5 for the Hybrid architecture.

4. **Interpretation boundary:** Peak-subset metrics pool timesteps labelled peak on **actual** validation CPU using train-fitted P90 thresholds (~25.65% of steps). Conclusions apply to this definition and protocol only.

---

### 3. Architecture sensitivity — why identical Peak-Aware methodology produced different outcomes

The locked experiments hold peak definition, λ, weighting granularity, and evaluation protocol constant. Observed divergence is therefore attributable to **architecture-specific behaviour**, not to different peak-aware hyperparameters.

**Evidence from completed experiments:**

| Factor | Hybrid track (negligible effect) | Global track (large trade-off) |
|--------|----------------------------------|--------------------------------|
| **Forecasting pipeline** | Prophet trend/seasonality + GRU **residual** correction | Direct **cpu_scaled** GRU forecasting (no Prophet) |
| **Training target** | `residual_scaled` — peaks reweighted only in residual learning path | `cpu_scaled` — peaks reweighted on full CPU target |
| **Baseline peak error** | Already lower (peak MAE 2.501) | Higher (peak MAE 3.426) — more headroom for peak-weighted training to reduce peak-subset error |
| **Peak-Aware Δpeak MAE** | +0.024 (worsened) | −0.777 (improved) |
| **Peak-Aware Δnon-peak MAE** | −0.015 (negligible) | +0.604 (+42.9%; worsened) |
| **Per-container overall** | Mixed (45 improved / 54 worsened) | Predominantly worse (23 improved / 76 worsened) |
| **Per-container peak** | Mixed peak outcomes | Peak improved on **89 / 98** containers with ≥1 peak step |

**Evidence-based explanation (not speculation beyond locked records):**

- On **Hybrid**, peak-period error in the final forecast depends on **both Prophet and GRU**. The locked Hybrid discussion notes that modifying only the GRU residual objective may not reduce peak CPU error if peak deviation is already captured or constrained by the Prophet component. The observed +0.024 peak MAE change is consistent with **no effective peak-subset benefit** from residual reweighting alone.

- On **Global**, the GRU models the **entire CPU signal** directly. Peak-weighted training can shift capacity toward high-utilization timesteps, producing a **large within-architecture peak gain** (−22.7% peak MAE vs Global Baseline). Because peak steps comprise ~26% of timesteps, aggressive up-weighting **degrades non-peak accuracy** (+42.9% non-peak MAE) and **overall accuracy** (+13.0% overall MAE). The locked Peak-Aware Global evaluation documents this Tier-1 / Tier-2 / Tier-3 split explicitly.

- **Same methodology, different error geometry:** Hybrid baselines already achieve lower overall and peak errors than Global baselines. Peak-Aware learning on Global mainly **compresses the Global peak gap toward Hybrid levels** (2.649 vs 2.501) while **sacrificing** Global's non-peak advantage (1.406 non-peak MAE on Global Baseline).

**Conclusion:** Peak-Aware learning is **strongly architecture-dependent**. It is **ineffective to marginally harmful** on Hybrid under this configuration, and **beneficial for peak timesteps only on Global**, with unacceptable overall and non-peak penalties for general deployment.

---

### 4. Production recommendations (evidence-calibrated)

| Scenario | Recommendation | Rationale (locked evidence) |
|----------|----------------|----------------------------|
| **General-purpose Day-1 CPU forecasting** | **Hybrid Prophet + GRU (Baseline)** | Best peak accuracy; overall performance statistically comparable to Peak-Aware Hybrid; simpler (no peak-weighted training); avoids negligible-gain replacement |
| **If Hybrid architecture is already deployed** | **Retain Hybrid Baseline** — do **not** migrate to Peak-Aware Hybrid for +0.005 MAE | Improvement is not operationally meaningful; peak MAE is slightly **worse** with Peak-Aware Hybrid (+0.024) |
| **Peak-timestep priority (best across all models)** | **Hybrid Baseline** | Lowest peak MAE (2.501) and peak RMSE (5.186) |
| **Global architecture required, peak priority only** | **Peak-Aware Global** (conditional) | Lowers peak MAE vs Global Baseline (3.426 → 2.649) but overall MAE is **highest among all four models** (2.174) and non-peak error increases substantially |
| **Non-peak accuracy, no Prophet** | **Global Baseline** | Lowest non-peak MAE (1.406); accept higher overall and peak error vs Hybrid |
| **Do not deploy as default** | **Peak-Aware Global** | Highest overall MAE; large non-peak degradation |

**Primary deployment conclusion:** The **Hybrid Prophet + GRU architecture** is the recommended general-purpose forecasting solution. Peak-Aware training **does not** provide sufficient evidence to replace Hybrid Baseline in production under the evaluated P90 + λ = 5 configuration.

---

## Thesis-ready answers (evidence-based)

### 1. Which model is best for general forecasting?

The **Hybrid Prophet + GRU architecture** (Baseline or Peak-Aware) provides the best overall performance. Hybrid Baseline (MAE 1.746) and Peak-Aware Hybrid (MAE 1.741) differ by **0.005 MAE (~0.3%)** — **comparable, not meaningfully different**. Both Hybrid models outperform Global Baseline (1.924) and Peak-Aware Global (2.174).

### 2. Which model is best for peak prediction?

**Hybrid Baseline** — peak MAE **2.501**, peak RMSE **5.186** (best among all four models). Peak-Aware Hybrid and Peak-Aware Global rank second and third on peak MAE; Global Baseline is worst (3.426).

### 3. Does Peak-Aware learning improve forecasting?

**It depends on architecture and metric tier.**

- **Hybrid:** No meaningful overall improvement (−0.005 MAE); **no peak-subset improvement** (+0.024 peak MAE). Peak-Aware learning **does not improve** forecasting under this configuration.
- **Global:** **Substantial peak-subset improvement** (−0.777 peak MAE) but **worse overall** (+0.249 MAE) and **worse non-peak** (+0.604 MAE). Peak-Aware learning **improves peak timesteps only** at the cost of other timesteps.

### 4. Is Peak-Aware learning architecture-dependent?

**Yes.** Identical P90 + λ = 5 methodology produced negligible Hybrid effects and a large Global peak/overall/non-peak trade-off (see architecture sensitivity table above).

### 5. Practical implications for cloud resource forecasting

For Alibaba trace–style containerized CPU forecasting under the locked 99-container Day-1 protocol: deploy **Hybrid Prophet + GRU (Baseline)** as the default forecaster; use **Global Baseline** only if Prophet overhead is unacceptable and non-peak accuracy is prioritised; treat **Peak-Aware Global** as a specialised, architecture-constrained option for peak-focused monitoring, not general provisioning; **do not adopt Peak-Aware Hybrid** over Hybrid Baseline given negligible aggregate benefit and slightly worse peak-subset performance.

---

## Final research conclusion

Under a locked fair-comparison protocol (99 containers, Day-1 horizon, P90 peak definition, λ = 5 timestep-weighted MSE), the **Hybrid Prophet + GRU architecture delivers the best overall and peak-subset forecasting performance** among all final models, with Hybrid Baseline and Peak-Aware Hybrid statistically comparable on overall error (ΔMAE 0.005). **Hybrid Baseline achieves the lowest peak-subset MAE (2.501)**, demonstrating that the evaluated Peak-Aware methodology **does not improve peak prediction on the Hybrid track** and **does not outperform the unmodified Hybrid model on peak timesteps across architectures**. On Global GRU, identical Peak-Aware training produces an **architecture-dependent trade-off**: large peak-subset gains (−0.777 MAE vs Global Baseline) accompanied by substantial non-peak and overall degradation, whereas Hybrid shows negligible movement on all tiers. **Peak-Aware learning is therefore architecture-dependent and is not a universal improvement** — it fails to deliver meaningful benefit on Hybrid and trades overall accuracy for peak accuracy on Global. For practical cloud resource forecasting, **Hybrid Baseline is the recommended general-purpose deployment**; Peak-Aware variants do not justify replacement of Hybrid Baseline on the evidence of this study.

---

## Provenance

All metrics were extracted read-only from locked experiment evaluation outputs. No models were retrained and no inference was regenerated. This document's **Interpretation revised** section refines academic wording only; numerical values are unchanged from `master_metrics_table.csv` and source experiments.

**Source experiments:**

| Model | Directory |
|-------|-----------|
| Hybrid Baseline | `experiments/baseline_reference_2026-07-14/` |
| Peak-Aware Hybrid | `experiments/peak_aware_2026-07-14_164518/` |
| Global Baseline | `experiments/global_gru_baseline_2026-07-17_121748/` |
| Peak-Aware Global | `experiments/peak_aware_global_validation_vs1_2026-07-17_170939/` |
