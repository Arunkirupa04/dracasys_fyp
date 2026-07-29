# Discussion

## Overview

Temporal Memory Analysis (TMA) was conducted to determine whether the fixed **96-step input window** used throughout this FYP was adequate for the temporal structure present in the selected Alibaba workloads. The answer is **not the same for Hybrid and Global GRU**, because those architectures receive **different input signals**.

This section explains what TMA proved, what it did not prove, and how it fits into the complete research story.

---

## Three distinct signals — never mix them

TMA deliberately analyzes three separate time series. Conflating them leads to incorrect conclusions.

### Signal 1: Original CPU

- **What it is:** Real CPU utilisation (%) from frozen train+validation parquets.
- **Who sees it:** Global GRU models forecast directly from scaled CPU sequences.
- **What TMA found:** Strong daily (lag-96 mean ACF ≈ 0.33) and multi-day (lag-192 ≈ 0.25) temporal memory. Only ~40% of cumulative squared-ACF energy is captured within 96 lags.

### Signal 2: Prophet-equivalent residual

- **What it is:** CPU minus train-fitted daily seasonality and linear trend (read-only Fourier OLS proxy for Prophet residuals).
- **Who sees it:** Hybrid GRU — the GRU **never receives original CPU**. It receives Prophet residuals during training and inference.
- **What TMA found:** Lag-96 mean ACF ≈ 0.04. Prophet removes most daily and multi-day temporal dependence before residual learning.

### Signal 3: Hybrid post-forecast residual

- **What it is:** `actual_day1_real − day1_final_real` from frozen baseline inference cache (96 validation steps).
- **What it represents:** Structured forecast error after the full Hybrid pipeline (Prophet + GRU), not the training target residual.
- **What TMA found:** Decorrelation within ~3 steps. Little long-range autocorrelation in day-1 forecast errors.

**Critical rule:** Window-sufficiency conclusions drawn from **original CPU** apply to **Global GRU**. They do **not** automatically transfer to **Hybrid GRU**, which operates on Signal 2.

---

## What TMA proved

### 1. Raw Alibaba workloads contain substantial long-range temporal memory

On **original CPU** (Signal 1):

- Lag-96 mean ACF ≈ **0.33** (62.6% of containers with |ACF| ≥ 0.1)
- Lag-192 mean ACF ≈ **0.25** (51.5% still ≥ 0.1 at 2 days)
- Cumulative squared-ACF capture within 96 lags: **40.0%**; beyond 96 lags: **60.0 percentage points**
- Multi-day periodic echoes persist through lags 288–480

This is quantitative proof that the selected workloads are **not memoryless** and that a one-day window on **raw CPU** does not encompass most autocorrelation energy.

### 2. Prophet removes most long-range dependence before Hybrid residual learning

On **Prophet-equivalent residuals** (Signal 2):

- Lag-96 mean ACF drops from 0.33 → **0.04**
- Daily-cycle PSD fraction ≈ **0.7%** (vs substantial daily power in raw CPU)

Prophet's role in the Hybrid pipeline — modelling trend and daily seasonality so the GRU can focus on residual correction — is **supported by the data**.

### 3. Hybrid forecast errors show minimal long-range structure

On **Hybrid post-forecast residuals** (Signal 3):

- Median decorrelation lag (|ACF| < 0.1): **~3 steps**
- Within the available 96-step day-1 block, post-forecast errors are near-uncorrelated

This indicates the Hybrid model is not systematically leaving exploitable long-lag error patterns on the day-1 horizon.

### 4. The 96-step window question has architecture-specific answers

| Architecture | Input signal | TMA finding | Implication |
|--------------|-------------|-------------|-------------|
| **Global GRU** | Original CPU | 96 lags capture ~40% of squared-ACF energy | Window may omit meaningful temporal information; longer windows (192, 288, 384) are a **scientifically justified** future direction |
| **Hybrid GRU** | Prophet residual | Lag-96 ACF ≈ 0.04; short-memory signal | 96-step residual window is **methodologically supported**; evidence does **not** support "Hybrid failed because window was too short" |

---

## What TMA did NOT prove

TMA is a **diagnostic on available autocorrelation structure**. It is not a forecasting experiment. The following claims are **explicitly outside TMA scope**:

| TMA does NOT claim | Why |
|--------------------|-----|
| Longer windows improve MAE | No models were retrained or evaluated under alternative window lengths |
| Longer windows improve RMSE | Same — no forecast comparison was conducted |
| Hybrid should use longer windows | Hybrid GRU does not see raw CPU; residual memory is short |
| Global GRU accuracy would necessarily improve | Autocorrelation presence ≠ forecastability by a given architecture |
| GRU failed because of insufficient context | CSRLE showed GRU *can* learn structure; LFHE showed dispersion issues; RRA Audit showed remaining structure is largely linear |

TMA only quantifies **how much temporal information is present** at different window lengths in each signal. Whether a model can **exploit** that information is a separate question requiring retraining experiments.

---

## Global GRU interpretation

Global GRU models in this project receive **scaled CPU sequences** directly. TMA Signal 1 therefore applies directly to Global forecasting.

**Evidence:**

- Lag-96 mean ACF ≈ 0.33
- Lag-192 mean ACF ≈ 0.25
- Only ~40% of cumulative squared-ACF energy within 96 lags
- Significant multi-day ACF at lags 192–480

**Conclusion:** TMA provides **scientific evidence** that the fixed 96-step input window may omit meaningful temporal information when forecasting directly from raw CPU. This **justifies future work** investigating longer windows such as **192, 288, and 384** steps for Global GRU models.

**Explicit caveat:** TMA does **not** prove that longer windows would improve forecasting accuracy. That requires controlled retraining and evaluation experiments.

---

## Hybrid GRU interpretation

The Hybrid pipeline is:

```
Original CPU
    ↓
Prophet  (trend + daily seasonality)
    ↓
Prophet residual
    ↓
GRU  (residual correction)
    ↓
Final forecast = Prophet + GRU correction
```

The Hybrid GRU **never receives original CPU**. It receives **Prophet residuals** (Signal 2).

**Evidence on the residual signal:**

- Lag-96 mean ACF ≈ **0.04** (vs 0.33 on raw CPU)
- Daily-cycle power largely removed (PSD fraction ≈ 0.7%)
- Residual presented to the GRU contains **much shorter memory** than raw CPU

**Conclusion:** The evidence from TMA does **not** support the hypothesis that the Hybrid GRU underperformed because the 96-step input window was too short. Prophet already removed most long-range temporal dependence. The residual signal the GRU sees is predominantly **short-memory**.

TMA therefore **supports the original Hybrid design**:

- **Prophet** models long-range temporal behaviour (trend, daily seasonality).
- **GRU** models short-term residual corrections on a de-seasonalised signal.

Hybrid forecasting limitations identified elsewhere in this project (residual learnability in CSRLE, variance collapse in LFHE, linear remaining structure in RRA Audit) should **not** be re-attributed to insufficient input-window length on raw CPU.

**Signal 3 (Hybrid post-forecast residual)** adds that day-1 forecast errors decorrelate within ~3 steps, consistent with a pipeline that is not leaving strong structured errors on the evaluation horizon — within the limits of the 96-step cached series.

---

## Relationship to previous experiments

TMA is the **final diagnostic in a coherent chain**. Each prior experiment addressed a different layer of the forecasting problem. TMA adds the **input-window / temporal-memory layer** without contradicting earlier findings.

```
Residual Pattern Analysis
    ↓
    Prophet validation residuals still contain temporal structure
    (ACF, PACF, Ljung–Box rejections)
    ↓
CSRLE (Controlled Synthetic Residual Learnability Experiment)
    ↓
    GRU *can* learn injected synthetic temporal structure under controlled conditions
    (residual learnability is possible in principle)
    ↓
LFHE (Loss Function Hypothesis Experiment)
    ↓
    Variance collapse is partly objective-driven (DA-MSE fixes dispersion
    but not correlation/MAE) — a training objective issue, not window length
    ↓
Ridge Residual Analysis + Audit
    ↓
    After Prophet, remaining structure is predominantly linear;
    Ridge does not remove it; corrected remaining structure persists
    ↓
TMA (Temporal Memory Analysis)
    ↓
    Original CPU has long memory (daily + multi-day)
    Prophet removes most long memory from the residual signal
    Hybrid GRU receives short-memory residuals → 96-step window supported
    Global GRU sees raw CPU → 96-step window may be short
```

### Why these findings are consistent, not contradictory

| Prior finding | TMA reconciliation |
|---------------|-------------------|
| RPA: Prophet residuals contain structure | TMA confirms structure exists in residuals (lag-96 ACF ≈ 0.04 is non-zero), but at **much lower magnitude** than raw CPU (0.33). Structure ≠ long-range daily memory. |
| CSRLE: GRU can learn synthetic structure | TMA does not dispute GRU capacity. It shows the **real Alibaba residual signal** has shorter memory than raw CPU, so window-length is unlikely the binding constraint for Hybrid. |
| LFHE: Dispersion collapse under MSE | Orthogonal to window length. LFHE explains *how well* the GRU learns; TMA explains *what memory is available* in the input. |
| RRA Audit: Remaining structure is linear | Consistent with TMA: after Prophet, residual memory is weaker and shorter. Linear structure (AR-like) at short lags does not require a 192+ step window. |

TMA **narrows the search space** for future improvements:

- For **Hybrid:** focus on loss functions (LFHE), linear vs nonlinear residual modelling (RRA Audit), not longer raw-CPU windows.
- For **Global GRU:** longer-context input windows are a justified — but unproven — direction.

---

## Final research interpretation

The complete evidence chain supports the following narrative:

```
Original CPU
    ↓  (strong daily + multi-day temporal memory;
    ↓   96 lags capture only ~40% of squared-ACF energy)
    ↓
Prophet decomposition
    ↓  (removes most long-range dependence;
    ↓   lag-96 residual ACF ≈ 0.04)
    ↓
Hybrid GRU receives short-memory residuals
    ↓  (96-step residual window methodologically supported)
    ↓
Hybrid architecture is supported by the evidence
```

Separately, for **Global forecasting**:

```
Global GRU receives raw CPU directly
    ↓  (full long-range memory present)
    ↓
96-step window appears short relative to available memory
    ↓
Longer-context Global models (192, 288, 384) are a
scientifically justified future research direction
    ↓  (performance improvement NOT proven by TMA)
```

---

## Implications for thesis

> The Temporal Memory Analysis demonstrates that the selected Alibaba workloads contain substantial long-range temporal dependence in the original CPU series. However, the Prophet decomposition employed within the Hybrid architecture removes most of this long-range structure before residual learning. Consequently, the evidence supports the use of a 96-step residual-learning window within the Hybrid pipeline while simultaneously identifying longer-context raw CPU forecasting as a promising direction for future Global forecasting research.

### Examiner-facing summary

| Question | Answer |
|----------|--------|
| Was the 96-step window wrong for Hybrid? | **No evidence supports that.** Residual memory is short after Prophet. |
| Was the 96-step window potentially limiting for Global GRU? | **Yes, plausibly.** Raw CPU retains multi-day memory beyond 96 lags. |
| Does TMA prove longer windows help Global GRU? | **No.** Justifies investigation only. |
| Does TMA explain why Hybrid GRU struggled? | **No.** Points to residual learnability (CSRLE), loss function (LFHE), and linear structure (RRA Audit) — not window length. |

---

## Verdict (unchanged numerically)

**For Global GRU:** Evidence indicates non-trivial temporal dependence beyond 96 lags on original CPU; future work **may** investigate longer input windows (192, 288, 384) — **without claiming forecast improvement**.

**For Hybrid GRU:** The 96-step residual-learning window remains **methodologically supported** by TMA evidence. Longer raw-CPU windows are **not** indicated for the Hybrid path.
