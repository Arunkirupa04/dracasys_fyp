# Temporal Memory Analysis — Final Report

**Timestamp:** 2026-07-26_170052  
**Protocol:** tma_v1.0

---

## Three distinct signals

This report analyzes **three separate time series**. Results must not be mixed across signals.

| Signal | Series | Applies to |
|--------|--------|-----------|
| 1 | Original CPU | Global GRU (raw CPU forecasting) |
| 2 | Prophet-equivalent residual | Hybrid GRU input (after Prophet decomposition) |
| 3 | Hybrid post-forecast residual | Hybrid forecast error (actual − forecast, day-1) |

---

## Explicit answers

### Q1. How much temporal memory exists in the selected Alibaba workloads?

**Signal 1 (Original CPU):** Cohort integrated autocorrelation time (CPU) median = 0.24 steps; mean cumulative squared-ACF capture within 96 lags = 40.0%; lag-96 mean ACF = 0.329.

### Q2. Does meaningful temporal dependence exist beyond one day?

**Signal 1 (Original CPU):** At lag 192 (2 days), mean ACF = 0.2533, 51.5% of containers have |ACF|≥0.1; paired beyond-vs-within mean |ACF| difference = -0.0965 (Wilcoxon p = 5.697e-18).

### Q3. Are daily or multi-day cycles still present after Prophet?

**Signal 2 (Prophet-equivalent residual):** After Prophet-equivalent decomposition, lag-96 mean ACF = 0.0420; mean daily-cycle PSD fraction = 0.0072.

### Q4. Does the Hybrid residual retain long-range temporal dependence?

**Signal 2 (residual input):** Lag-96 ACF ≈ 0.04 — much weaker than raw CPU (0.33). Long-range daily dependence is largely removed before GRU residual learning.

**Signal 3 (post-forecast error):** Hybrid day-1 post-forecast residual decorrelation lag (|ACF|<0.1) median = 3.0 steps (96-step series); within-96 capture = 100.0% when defined.

### Q5. Was the chosen 96-step input window sufficient?

**Signal 1 (Global GRU — original CPU):** Mean cumulative squared-ACF capture: 96 lags = 40.0%, 672 lags = 100.0%; gap beyond 96 lags = 60.0 percentage points. The 96-step window captures a minority of autocorrelation energy on raw CPU.

**Signal 2 (Hybrid GRU — Prophet residual):** Lag-96 mean ACF = 0.042. Residual memory is short. The 96-step residual-learning window is methodologically supported.

### Q6. Would longer windows (192, 288, 384) provide additional temporal information?

**Signal 1 (Global GRU only):** Available mean |ACF| within window — 96: 0.1938, 384: 0.1430; increment 96→384 = -0.0507 (information proxy only, not forecast gain). Cumulative squared-ACF capture rises from 40% (96 lags) to 64% (192) to 79% (288), indicating additional autocorrelation energy exists beyond 96 lags on raw CPU.

**Signal 2 (Hybrid GRU):** Not applicable — residual signal already has short memory (lag-96 ACF ≈ 0.04).

### Q7. Should future work investigate longer input windows?

**Global GRU:** Evidence indicates non-trivial temporal dependence beyond 96 lags on original CPU; future work **may** investigate longer input windows (192, 288, 384) — **without claiming forecast improvement**.

**Hybrid GRU:** Evidence does **not** support investigating longer windows. Prophet removes most long-range structure; the GRU receives short-memory residuals. Hybrid limitations should be pursued through CSRLE, LFHE, and RRA Audit channels.

---

## What TMA proved

1. **Original CPU** (Signal 1) retains substantial daily and multi-day temporal memory beyond one day.
2. **Prophet-equivalent residuals** (Signal 2) contain much weaker long-range dependence — Prophet removes most daily temporal structure.
3. **Hybrid post-forecast errors** (Signal 3) decorrelate within ~3 steps on the day-1 block.
4. For **Global GRU**, the 96-step window may omit meaningful temporal information on raw CPU — longer windows are **scientifically justified** for future investigation.
5. For **Hybrid GRU**, the 96-step residual-learning window is **methodologically supported** — the evidence does not support "Hybrid failed because the window was too short."

## What TMA did NOT prove

TMA quantifies available temporal autocorrelation only. It does **not** claim:

- Longer windows improve MAE or RMSE
- Hybrid should use longer windows
- Global GRU accuracy would necessarily improve with longer context
- The GRU failed because of insufficient input context

---

## Global GRU interpretation

Global GRU models forecast directly from scaled CPU (Signal 1). TMA shows:

- Lag-96 mean ACF ≈ 0.33; lag-192 ≈ 0.25
- Only ~40% of cumulative squared-ACF energy within 96 lags

**Conclusion:** TMA provides scientific evidence that the fixed 96-step input window may omit meaningful temporal information for raw CPU forecasting. Investigating longer windows (192, 288, 384) is a **justified future research direction**. TMA does **not** prove that longer windows would improve forecasting accuracy.

---

## Hybrid GRU interpretation

The Hybrid pipeline is: Original CPU → Prophet → Residual → GRU. The Hybrid GRU **never receives original CPU**; it receives Prophet residuals (Signal 2).

TMA shows lag-96 residual ACF ≈ **0.04** (vs 0.33 on raw CPU). Prophet already removed most daily temporal dependence.

**Conclusion:** TMA does **not** support the hypothesis that Hybrid underperformed because the 96-step window was too short. TMA **supports the original Hybrid design**: Prophet models long-range behaviour; GRU models short-term residual corrections on a substantially shorter-memory signal.

---

## Relationship to previous experiments

| Experiment | Finding | TMA reconciliation |
|------------|---------|-------------------|
| Residual Pattern Analysis | Prophet residuals contain structure | Confirmed — but at much lower magnitude than raw CPU (0.04 vs 0.33) |
| CSRLE | GRU can learn synthetic temporal structure | Consistent — GRU capacity is not disputed; real residual memory is shorter |
| LFHE | Variance collapse is partly objective-driven | Orthogonal — LFHE addresses *how* GRU learns; TMA addresses *what memory is available* |
| Ridge Residual Audit | Remaining structure is predominantly linear | Consistent — short-memory linear structure does not require 192+ step windows |

TMA completes the diagnostic chain by adding the **input-window / temporal-memory layer** without contradicting prior findings.

---

## Final research interpretation

```
Original CPU  →  strong daily + multi-day memory (Signal 1)
       ↓
Prophet  →  removes most long-range dependence (Signal 2: lag-96 ACF ≈ 0.04)
       ↓
Hybrid GRU receives short-memory residuals → 96-step window supported
       ↓
Hybrid architecture is methodologically supported by the evidence
```

For Global forecasting separately:

```
Global GRU receives raw CPU  →  full long-range memory present
       ↓
96-step window appears short (~40% squared-ACF capture)
       ↓
Longer-context Global models are a scientifically justified future direction
(not performance-proven)
```

---

## Implications for thesis

> The Temporal Memory Analysis demonstrates that the selected Alibaba workloads contain substantial long-range temporal dependence in the original CPU series. However, the Prophet decomposition employed within the Hybrid architecture removes most of this long-range structure before residual learning. Consequently, the evidence supports the use of a 96-step residual-learning window within the Hybrid pipeline while simultaneously identifying longer-context raw CPU forecasting as a promising direction for future Global forecasting research.

---

## Limitations

- Prophet residuals use read-only Fourier daily+trend OLS (train-fitted), not Prophet.fit().
- Hybrid post-forecast residual limited to frozen validation day-1 (96 steps) from baseline inference cache.
- Long-lag ACF/PACF at 672 lags applies to CPU and Prophet-equivalent residuals on full train+val timeline (~768 steps).
- Hypothetical window analysis quantifies available autocorrelation only; it does not estimate forecast accuracy gains.
- Three signals must be interpreted independently; CPU long-memory does not imply Hybrid needs longer windows.

---

## Documentation

Full interpretation: `docs/temporal_memory_analysis/discussion.md`  
Thesis summary: `docs/temporal_memory_analysis/summary.md`
