# Background

## Research chain position

| Study | Relevance to GGTCE |
|-------|-------------------|
| **Global GRU baseline (G96)** | Frozen control — direct CPU, 96×3 features → 96 output |
| **TMA** | **Primary justification** — raw CPU retains long memory; 96-step window captures ~40% squared-ACF energy |
| **Hybrid / HCERL** | Longer context **not** supported for Prophet residuals; isolates Global as the correct testbed |
| **CSRLE / LFHE** | GRU can learn structure when present; objective affects dispersion — monitor but do not change loss |

## TMA key finding (Signal 1 — original CPU)

From `experiments/temporal_memory_analysis_2026-07-26_170052`:

| Metric | Value |
|--------|-------|
| Lag-96 mean ACF | **0.33** |
| Lag-192 mean ACF | **0.25** |
| Mean squared-ACF capture @ 96 lags | **40.0%** |
| Mean squared-ACF capture @ 192 lags | **59.9%** |
| Mean squared-ACF capture @ 288 lags | **73.4%** |
| Beyond-96 capture gap | **60.0 pp** |

TMA explicitly states: *"96-step window may be short for Global GRU on raw CPU"* — but **does not prove forecast improvement**.

## Why Hybrid is excluded from this experiment

Prophet removes daily/multi-day structure before Hybrid GRU sees residuals (lag-96 residual ACF ≈ 0.04). HCERL confirmed context channels do not help Hybrid on real data. GGTCE targets the **architectural path that still observes raw CPU history**.

## Global baseline today

```
Input:  (96, 3)  — cpu_scaled, cpu_mean, cpu_std
Output: (96,)    — cpu_scaled Day-1 forecast
```

Only **input_window** is proposed to change.

## Dataset context

- 99 evaluable containers (100 selected; `c_14674` skipped)
- ~600–614 train steps per container (~6.3 days @ 15 min)
- ~150–154 validation steps per container
- Hard upper bound on input window: `n_train - forecast_horizon - 1` ≈ **517 steps** maximum in principle; practically less for stable sliding windows
