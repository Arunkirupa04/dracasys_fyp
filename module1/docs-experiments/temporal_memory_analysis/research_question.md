# Research Question

## Primary question

> Did the selected 96-step input window capture all meaningful temporal dependencies present in the selected Alibaba workloads, or does useful memory exist beyond one day?

## Scope clarification: two architectures, three signals

This question must be answered **separately** for each forecasting architecture, because they receive **different input signals**:

| Architecture | Input signal | TMA signal analyzed |
|--------------|-------------|---------------------|
| **Global GRU** | Scaled original CPU | Signal 1: Original CPU |
| **Hybrid GRU** | Prophet residual (after decomposition) | Signal 2: Prophet-equivalent residual |
| **Hybrid evaluation** | Forecast error | Signal 3: Hybrid post-forecast residual |

**Do not mix signals.** Long-memory findings on original CPU (Signal 1) inform Global GRU window-length questions. They do **not** automatically apply to Hybrid GRU, which never sees raw CPU.

## Context

Throughout this FYP, Hybrid and Global GRU models used:

| Parameter | Value |
|-----------|-------|
| Input window | 96 timesteps (1 day) |
| Output window | 96 timesteps (1 day) |
| Sampling | 15 minutes |
| Cohort | 99 evaluable validation containers |

The preprocessed Alibaba trace spans approximately **7 days** per container (train + validation chronology).

TMA was designed to close a methodological gap: before concluding that the GRU architecture itself is the limiting factor, the project needed evidence on whether the fixed 96-step window was adequate for the temporal structure present in the data — **for each architecture's actual input signal**.

## Sub-questions (final report)

1. How much temporal memory exists in the selected Alibaba workloads? *(Signal 1 — CPU)*
2. Does meaningful temporal dependence exist beyond one day? *(Signal 1 — CPU)*
3. Are daily or multi-day cycles still present after Prophet? *(Signal 2 — residual)*
4. Does the Hybrid residual retain long-range temporal dependence? *(Signals 2 and 3)*
5. Was the chosen 96-step input window sufficient? *(Signal 1 for Global; Signal 2 for Hybrid)*
6. Is there evidence that longer windows (192, 288, 384) would provide additional temporal information? *(Signal 1 for Global only)*
7. Based on evidence only, should future work investigate longer input windows? *(Global: yes, scientifically justified; Hybrid: no, not supported)*

## Success criteria

- All conclusions supported by quantitative diagnostics (ACF/PACF, FFT, window sufficiency, paired tests).
- Three signals interpreted independently.
- Global vs Hybrid implications stated explicitly.
- No forecast performance claims from window hypotheticals.
- No model retraining.

## Position in the research narrative

TMA follows Residual Pattern Analysis, CSRLE, LFHE, and Ridge Residual Audit in the diagnostic chain. It adds the **temporal memory / input-window layer** without contradicting prior findings. See [discussion.md](discussion.md#relationship-to-previous-experiments).
