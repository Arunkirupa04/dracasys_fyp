# Analysis Pipeline

## Overview

```
Frozen Alibaba data (train + val parquets, scalers, inference cache)
        ↓
Per-container series:
  • Signal 1 — CPU (real %)           → Global GRU relevance
  • Signal 2 — Prophet-equiv. residual → Hybrid GRU input relevance
  • Signal 3 — Hybrid post-forecast    → Hybrid forecast-error relevance
        ↓
Long-lag ACF / PACF (≤ 672)   [Signals 1 & 2; Signal 3 capped at 95 lags]
        ↓
Daily periodicity at lags 96, 192, …, 672
        ↓
FFT / PSD
        ↓
Memory length estimation
        ↓
Window sufficiency + hypothetical windows
        ↓
Paired statistics (within day 1 vs beyond)
        ↓
Cohort aggregation + case studies + final report
```

## Cohort aggregation

For each series type and lag axis:

- **Mean** and **median** ACF/PACF across 99 containers
- **95% CI** on cohort mean (normal approximation of container-level ACF values)

## Memory length metrics (per container)

| Metric | Definition |
|--------|------------|
| First insignificant lag | First lag where \|ACF\| < 1.96/√n |
| Half-life | First lag where ACF < 0.5 |
| Decorrelation (0.1) | First lag where \|ACF\| < 0.1 |
| Integrated autocorrelation time | τ = 1 + 2 Σ ACF(k) |

## Window sufficiency

Cumulative squared-ACF capture:

\[
\text{capture}(W) = \frac{\sum_{k=1}^{W} \text{ACF}(k)^2}{\sum_{k=1}^{K} \text{ACF}(k)^2}
\]

where K is the effective maximum lag for the series.

## Case study selection

Based on CPU integrated autocorrelation time:

- Strongest long-memory container
- Median container
- Weakest long-memory container

## Code entry point

```bash
.venv/bin/python scripts/run_temporal_memory_analysis.py
```

## Interpretation note

Each diagnostic output is tagged to one of three signals. Cohort aggregation is performed **per signal, per series type** — results are never pooled across signals. See [methodology.md](methodology.md) and [discussion.md](discussion.md) for architecture-specific interpretation (Global vs Hybrid).
