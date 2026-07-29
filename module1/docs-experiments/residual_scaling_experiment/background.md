# Background

## Forecasting pipeline

The Hybrid model decomposes CPU utilisation into:

1. **Prophet** — captures ~78% of variance via daily seasonality (per-container, train-fit).
2. **GRU** — models the Prophet complement residual over a 96-step input window, predicting 96 future residual steps (Day-1 horizon).

## Prior experimental evidence

| Experiment | Question | Outcome |
|------------|----------|---------|
| Baseline Hybrid | Does GRU help? | Yes — consistent improvement over Prophet |
| Peak-aware loss | Does reweighting peaks help? | No significant gain |
| HCERL | Does context enrichment help? | No significant gain |
| GGTCE / TMA | Does window length matter? | 96-step window sufficient |
| Stage 0 RRE diagnostics | Which representation? | R1 rejected; R3 same temporal info as R0 |

## Implication

Training-side and window-side changes have been exhausted under the frozen protocol. The unresolved question shifts to **how the residual is scaled before GRU ingestion** — an optimisation and numerical stability concern, not a representation of new temporal dynamics.

## Stage 0 reference

`experiments/rre_diagnostics_2026-07-27_120759`

Key numbers (train cohort, 99 containers):

- R0 mean |ACF| (lags 1–96): 0.0348
- R1 mean |ACF|: 0.0116 (rejected)
- R3 mean |ACF|: 0.0348 (identical to R0)
- R3 sparsity (|x| < 0.05): 3.3% vs R0 10.5%
