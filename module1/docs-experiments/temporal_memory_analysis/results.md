# Results

**Authoritative run:** `experiments/temporal_memory_analysis_2026-07-26_170052/`

## How to read these results

TMA reports outcomes for **three separate signals**. Each signal answers a different architectural question. **Do not combine or average across signals.**

| Signal | Applies to | Window-sufficiency question |
|--------|-----------|----------------------------|
| Original CPU | Global GRU | Is 96 steps enough context for raw CPU forecasting? |
| Prophet-equivalent residual | Hybrid GRU input | Is 96 steps enough for residual learning after Prophet? |
| Hybrid post-forecast residual | Hybrid forecast quality | Does the Hybrid model leave structured forecast errors? |

---

## Cohort summary

| Series | Capture @ 96 lags | Capture @ 672 lags | Lag-96 mean ACF | Lag-192 mean ACF | Integrated τ median |
|--------|-------------------|--------------------|-----------------|------------------|---------------------|
| Original CPU | 40.0% | 100% | 0.329 | 0.253 | 0.24 steps |
| Prophet-equivalent residual | 49.3% | 100% | 0.042 | −0.026 | 0.49 steps |
| Hybrid post-forecast (day-1) | 100%* | 100%* | N/A† | N/A† | ~0 steps |

\*Hybrid series is 96 steps; capture metrics saturate by construction.  
†Daily lags exceed hybrid series length.

---

## Signal 1 — Original CPU (Global GRU relevance)

### Q1 — Temporal memory magnitude

- Integrated autocorrelation time median: **0.24 steps** (this metric understates periodic memory; see limitations).
- Cumulative squared-ACF capture within 96 lags: **40.0%** of full-lag energy.
- Lag-96 (1-day) mean ACF: **0.329** (62.6% of containers with |ACF| ≥ 0.1).

**Interpretation:** Raw Alibaba workloads contain substantial temporal memory, dominated by daily and multi-day periodic structure.

### Q2 — Dependence beyond one day

| Lag | Days | Mean ACF | % \|ACF\| ≥ 0.1 |
|-----|------|----------|----------------|
| 96 | 1 | 0.329 | 62.6% |
| 192 | 2 | 0.253 | 51.5% |
| 288 | 3 | 0.187 | 49.5% |
| 384 | 4 | 0.164 | 49.5% |
| 480 | 5 | 0.131 | 49.5% |
| 576 | 6 | 0.085 | 40.4% |
| 672 | 7 | 0.040 | 30.3% |

Paired test (mean |ACF| lags 1–96 vs beyond 96):

| Metric | Value |
|--------|-------|
| Mean within (1–96) | 0.194 |
| Mean beyond (>96) | 0.097 |
| Difference (beyond − within) | −0.097 |
| Wilcoxon p | 5.7 × 10⁻¹⁸ |
| Cohen's d_z | −1.13 |

Short-lag |ACF| is stronger on average, but **multi-day periodic echoes** remain significant at lags 192–480.

**Interpretation (Global GRU):** TMA provides scientific evidence that the 96-step input window may omit meaningful temporal information when forecasting from raw CPU. Longer windows (192, 288, 384) are a **justified future research direction**. TMA does **not** prove they would improve MAE or RMSE.

### Q5 — Window sufficiency (CPU only)

| Window | Mean cumulative squared-ACF capture |
|--------|-------------------------------------|
| 96 lags | 40.0% |
| 192 lags | 63.9% |
| 288 lags | 78.5% |
| 672 lags | 100% |

**60 percentage points** of squared-ACF energy lie beyond lag 96 on original CPU.

### Q6 — Hypothetical longer windows (CPU only; information proxy)

Mean |ACF| averaged over lags 1..W:

| Window W | Mean \|ACF\| within window |
|----------|----------------------------|
| 96 | 0.194 |
| 192 | 0.168 |
| 288 | 0.155 |
| 384 | 0.143 |

This metric dilutes as W grows; cumulative capture (Q5) is the primary window-sufficiency indicator. These values quantify **available autocorrelation**, not forecast accuracy.

---

## Signal 2 — Prophet-equivalent residual (Hybrid GRU input relevance)

### Q3 — Daily cycles after Prophet-equivalent decomposition

- Lag-96 mean ACF on Prophet-equivalent residual: **0.042** (vs 0.329 on raw CPU).
- Mean daily-cycle PSD fraction: **0.007** (≈0.7% of non-DC power near f = 1/96).

**Interpretation (Hybrid GRU):** Prophet removes most daily and multi-day temporal structure. The residual signal the Hybrid GRU receives has **much shorter memory** than raw CPU. Lag-96 ACF of 0.04 vs 0.33 on CPU is the central Hybrid finding.

**This does NOT support** the hypothesis that Hybrid underperformed because the 96-step window was too short on raw CPU — the GRU never sees raw CPU.

Prophet-equivalent decomposition removes most predictable daily structure, aligning with the Hybrid design:

```
Prophet  →  long-range temporal behaviour (trend + daily seasonality)
GRU      →  short-term residual corrections on a de-seasonalised signal
```

---

## Signal 3 — Hybrid post-forecast residual (Hybrid forecast quality)

### Q4 — Long-range dependence in Hybrid forecast errors

- Decorrelation lag (|ACF| < 0.1) median: **3 steps** on 96-step day-1 series.
- Hybrid post-forecast errors show **minimal long-range autocorrelation** within the forecast horizon block.

**Interpretation:** The Hybrid pipeline (Prophet + GRU) does not systematically leave strong structured errors on the day-1 validation horizon — within the limits of the 96-step cached series. This is a **forecast-error diagnostic**, not an input-window diagnostic.

---

## Q7 — Recommendation (architecture-specific)

### Global GRU

Evidence supports **investigating** longer input windows in future work:

1. Only **40%** of squared-ACF energy is within 96 lags on **original CPU**.
2. **51.5%** of containers still have |ACF| ≥ 0.1 at lag 192.
3. Multi-day lag peaks remain through lag 480.

Candidate windows: **192, 288, 384** steps.

**Explicit non-claim:** This is **not** a claim that longer windows would improve Hybrid or Global GRU forecast accuracy.

### Hybrid GRU

TMA does **not** recommend longer windows for Hybrid:

1. Residual lag-96 ACF ≈ **0.04** — short-memory input signal.
2. Prophet already handles long-range structure.
3. 96-step residual-learning window is **methodologically supported**.

Hybrid limitations should be investigated through prior diagnostic channels (CSRLE, LFHE, RRA Audit), not through longer raw-CPU windows.

---

## Case studies

Case studies are selected on **CPU integrated autocorrelation time** (Signal 1 only).

| Role | Container |
|------|-----------|
| Strongest long-memory | c_12231 |
| Median | c_10132 |
| Weakest long-memory | c_14471 |

Figures: `plots/case_study_*_c_*.png`

---

## What these results do not show

See [limitations.md](limitations.md) and [discussion.md](discussion.md):

- TMA does not compare MAE/RMSE under different window lengths.
- TMA does not prove Global GRU would improve with longer windows.
- TMA does not recommend longer windows for Hybrid.
- TMA does not claim the GRU failed due to insufficient context.
