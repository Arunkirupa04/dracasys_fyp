# Summary

**Protocol:** `hcerl_v1.0`  
**Authoritative run:** `experiments/hcerl_2026-07-26_180649/`

## One-sentence takeaway

Adding context features to the Hybrid GRU **does not improve** Day-1 CPU forecasting on the Alibaba cohort; **V0 (residual-only) remains best** on MAE and residual Pearson r.

## Explicit answers

1. **Does context improve Hybrid forecasting?** — **No clear improvement** (V0 best MAE 1.732 vs V4 1.752).
2. **Most contributing feature?** — `is_peak_p90` (only feature with negative Δ MAE, but not significant).
3. **Least contributing feature?** — `cpu_std` (largest MAE degradation +0.011 pp).
4. **Prophet vs statistical?** — Prophet level ranked higher on composite; neither improved MAE vs V0.
5. **Peak features help?** — Marginal Pearson r gain V3→V4 (+0.020 mean); MAE −0.005 pp (not significant).
6. **Practically meaningful?** — **No** (best variant MAE Δ vs V0 = +0.48% for V4, worse).
7. **Replace baseline?** — **No — keep V0.**

## Recommendation

**Retain Hybrid V0 (96×1 residual-only).** Do not deploy V4 despite extra complexity.

## Thesis-safe statement

> The HCERL ablation tested cumulative context channels justified by the feasibility study. On real Alibaba data, none produced statistically significant Day-1 CPU improvements; the residual-only baseline achieved the lowest cohort MAE and highest residual Pearson r. Context-enriched Hybrid v2 is **not supported** for this dataset under frozen MSE training.
