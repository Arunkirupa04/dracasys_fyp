# Residual Pattern Analysis Documentation

> **This file has been superseded by the full notebook findings document:**
>
> **[`docs/residual-pattern-analysis-findings.md`](./residual-pattern-analysis-findings.md)**
>
> That document includes **all 16 sections** of `notebooks/residual_pattern_analysis.ipynb` (setup, Prophet diagnostics, ACF/PACF/Ljung–Box, distribution, frequency analysis, GRU quality, Conclusion C, **Hybrid post-forecast white-noise diagnostics**) **plus** the complete Section 13 evaluation audit.

---

## Quick reference — Section 13 audit verdict

**Verdict:** **A — Evaluation is correctly implemented; low correlation reflects genuine model behaviour**

| Question | Answer |
|----------|--------|
| Correct frozen GRU? | Yes — `experiments/baseline_reference_2026-07-14/models/hybrid_gru.keras` |
| Same residual space for actual vs predicted? | Yes — raw scaled residual (`cpu_scaled − prophet`) |
| Timestamps aligned? | Yes — Day-1, 96 steps |
| Hybrid reconstruction matches official pipeline? | Yes — max diff = 0.0 |
| Why is Pearson r ≈ 0.02? | GRU outputs near-constant corrections (~30× lower variance than actual residuals) |

---

## Quick reference — Sections 15–16 (Hybrid post-forecast white noise)

**Question:** After Hybrid, do final errors (`Actual − Hybrid`) look more like white noise than Prophet-only on the same Day-1 window?

**Verdict:** **Mostly no — substantial structure remains**

| Metric (Day-1, same window) | Prophet Day-1 | Hybrid Day-1 |
|-----------------------------|---------------|--------------|
| Ljung–Box reject rate (lag 20) | 56.6% | **54.5%** (−2.1 pp) |
| Average \|ACF\| lags 1–10 | 0.1889 | **0.1899** (unchanged) |
| Average \|PACF\| lags 1–10 | 0.1146 | 0.1135 |

- **54.5%** of containers still reject white noise after Hybrid — the GRU did not whiten Prophet errors.
- The tiny Ljung–Box improvement at lag 20 is consistent with Section 13’s near-flat corrections (bias shift, not dynamic modelling).
- Do **not** compare 73.7% (Section 14, full validation) to 54.5% (Hybrid Day-1) without noting the shorter 96-step window.

See full interpretation, comparison tables, and cross-section synthesis in [`residual-pattern-analysis-findings.md`](./residual-pattern-analysis-findings.md) (Sections 15–16).
