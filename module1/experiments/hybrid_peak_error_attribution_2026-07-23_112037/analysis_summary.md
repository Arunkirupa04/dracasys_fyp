# Hybrid Peak Error Attribution — Post-Hoc Diagnostic Summary

**Generated:** 2026-07-23T11:21:04Z  
**Label:** Post-hoc diagnostic analysis — NOT a new experiment.  
**Purpose:** Supporting evidence for thesis discussion on architecture-dependent Peak-Aware behaviour.

## Method

- Cohort: 99 evaluable containers (same as locked Peak-Aware Hybrid evaluation)
- Peak definition: train-fitted P90 thresholds from locked Peak-Aware Hybrid experiment
- Hybrid baseline arrays: verified against locked `baseline_inference_cache.pkl`
- Prophet component: extracted via frozen `run_hybrid_inference()` (not stored in locked cache)
- Locked experiment artifacts: **not modified**

## Pooled cohort averages (containers with ≥1 peak step where applicable)

| Metric | Value |
|--------|-------|
| Containers analysed | 99 |
| Containers with ≥1 peak step | 98 |
| Pooled Prophet Peak MAE | 2.4824 |
| Pooled Hybrid Peak MAE | 2.5012 |
| Pooled mean residual correction at peaks | 0.0779 |
| Pooled peak error reduction (Prophet − Hybrid) | -0.0188 |
| Pooled contribution ratio | -0.0076 |
| Mean per-container contribution ratio | -0.0030 |
| Median per-container contribution ratio | -0.0020 |

## Peak error reduction distribution

- Mean: -0.0119
- Median: -0.0036
- Std: 0.0717

## Verification against locked evaluation

- Status: **PASS**
- Hybrid peak MAE max abs delta vs locked CSV: 4.44e-16
- Hybrid final array max abs delta vs locked cache: 0.00e+00

## Diagnostic questions (evidence-based wording)

### 1. Does Prophet already explain most peak behaviour?

The evidence suggests Prophet already accounts for a substantial share of peak-period accuracy. Pooled Prophet Peak MAE is **2.482** versus pooled Hybrid Peak MAE **2.501** (Hybrid/Prophet ratio ≈ **1.008**). The pooled contribution ratio is **-0.008** (fractional reduction in peak MAE attributable to the GRU path). Per-container peak error reduction has mean **-0.012** and median **-0.004**. This is consistent with Prophet explaining most peak behaviour under this decomposition, without claiming Prophet is solely responsible for peak accuracy.

### 2. How much does the GRU reduce peak error?

At the pooled cohort level, the GRU residual path produces a **small net increase** in peak error: **0.019** MAE (Prophet Peak MAE **2.482** − Hybrid Peak MAE **2.501**). Per-container peak error reduction: mean **−0.012**, median **−0.004**, std **0.072**. The pooled contribution ratio (**−0.008**) indicates the GRU correction is **negligible relative to Prophet peak error** and does not materially improve—and may slightly worsen—peak-subset accuracy at the cohort level.

### 3. Is the residual correction during peaks generally small or large?

The mean absolute residual correction at peak timesteps is **0.078** CPU percentage points (pooled), versus Prophet Peak MAE **2.482**. The correction magnitude is **small relative to Prophet peak error** (ratio ≈ **0.031**). Per-container median correction: **0.029**. The results are consistent with peak-period forecasts being driven primarily by the Prophet component, with relatively small GRU adjustments at labelled peak steps.

### 4. Does the evidence support the interpretation that Peak-Aware learning has limited influence because the GRU only models residuals?

The findings support the interpretation that Peak-Aware GRU reweighting has **limited room to influence final peak forecasts** on the Hybrid architecture: pooled peak error change attributable to the GRU is **−0.019** MAE (pooled contribution ratio **−0.008**; GRU slightly worsens pooled peak accuracy vs Prophet-only). Because Peak-Aware training modifies only the **residual** objective—and the diagnostic shows residual corrections at peaks are comparatively small (**0.078** mean |Hybrid−Prophet|)—the evidence is **consistent with** the locked Peak-Aware Hybrid result (peak MAE +0.024 vs baseline). This does **not** prove causation; it provides decomposition evidence aligned with the architecture sensitivity discussion.

## Scope boundary

This diagnostic describes association and decomposition under the locked protocol. 
It does **not** establish causal mechanisms. Wording is intentionally cautious.
