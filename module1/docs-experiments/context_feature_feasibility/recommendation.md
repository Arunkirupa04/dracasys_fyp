# Recommendation

## Decision

**Proceed to a controlled Hybrid v2 feasibility experiment** — but with **modest expectations** and a **compact feature set**.

A context-enriched Hybrid is **scientifically justified as an investigatory branch**, not as a presumed fix for Hybrid underperformance.

---

## Why investigate (evidence chain)

| Finding | Implication for context features |
|---------|----------------------------------|
| **RPA:** 73.7% containers reject residual white noise | Signal exists post-Prophet |
| **CSRLE:** GRU learns synthetic structured residuals | Architecture *can* exploit temporal structure when signal is clear |
| **LFHE:** Dispersion recovers without correlation gain | GRU may need **regime/volatility cues** + better objective, not longer windows |
| **TMA:** Residual lag-96 ACF ≈ 0.04; 96-step window OK | Don't add calendar/long-lag features; **level/volatility** is the gap |
| **RLLA:** Real residual linear r ≈ 0.02–0.06 | True signal is **weak** — any gain will be incremental |
| **Ridge audit:** Linear model beats GRU on correlation | GRU optimization/generalization failure, not missing linear lags |

**Synthesis:** The Hybrid GRU receives **only** correction history. Prophet removes seasonality but **level-dependent correction behaviour** (peaks, heteroscedasticity) is not explicit in residual-only input. Context features targeting **operating level** and **local volatility** address a plausible representational gap — without contradicting TMA's window-length conclusion.

---

## What not to do

- Do **not** expand to 15+ features
- Do **not** rely on mem/net/disk telemetry
- Do **not** expect calendar features to help materially
- Do **not** extend input window for Hybrid (TMA does not support it)
- Do **not** treat high residual-derived correlation as justification for rolling-mean/lag features

---

## Recommended feature set (Hybrid v2-full)

| Channel | Feature | Rationale |
|---------|---------|-----------|
| 0 | `residual_scaled` | Baseline |
| 1 | `prophet_yhat_scaled` | Explicit operating level on seasonal surface |
| 2 | `res_roll_std_12` | Local volatility / heteroscedasticity |
| 3 | `cpu_std` | Container dispersion prior (Global GRU precedent) |
| 4 | `is_peak_p90` | High-load regime (peak-aware precedent) |

**Input shape:** `(96, 5)`

---

## Recommended experimental protocol (future — not this stage)

1. **Incremental ablation:** 96×1 → 96×2 → 96×3 → 96×5
2. **Metrics:** MAE, RMSE, MAPE (CPU level) + residual Pearson r, std ratio, Ljung–Box (diagnostic)
3. **Guardrail:** Reject v2 if MAE worsens >1% without r improvement (LFHE lesson)
4. **Same cohort:** 100 selected containers; frozen Prophet + splits
5. **Optional Phase 2:** v2-best + DA-MSE if dispersion collapse persists

---

## Go / no-go criteria

| Criterion | Go | No-go |
|-----------|-----|-------|
| Dataset availability | ✅ CPU-side features exist | Multivariate required |
| Leakage control | ✅ Train-only stats | — |
| Scientific non-contradiction with TMA | ✅ Level/vol, not long calendar | Longer window primary |
| Expected effect size | Incremental | Breakthrough |
| Implementation cost | Low–medium | — |

**Current status:** **GO** for investigatory Hybrid v2 experiment with compact features.

---

## Final recommendation (one paragraph)

Investigate Hybrid v2 with **five input channels**: scaled residual (existing), scaled Prophet level, 12-step rolling residual volatility, container `cpu_std`, and train-P90 peak indicator. This targets **regime and scale context** that Prophet + residual history do not make explicit, aligns with LFHE and peak-aware findings, and respects TMA's conclusion that calendar/long-memory features are redundant for Hybrid. Do **not** implement a broad feature stack. Treat outcome as **hypothesis confirmation or refutation** — RLLA and LFHE imply limited upside unless combined with loss-function changes.
