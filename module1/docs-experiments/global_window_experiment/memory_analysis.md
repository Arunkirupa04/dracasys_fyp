# GGTCE Memory-Aware Analysis

Co-primary objective: relate window-length benefit to Temporal Memory Analysis (TMA).

## Data sources

- TMA run: `experiments/temporal_memory_analysis_2026-07-26_170052/`
- Window sufficiency JSON (pct_capture_within_96/192/288)
- Memory length JSON (integrated_autocorr_time, decorrelation_lag)
- Per-container ACF at lags 96, 192, 288 recomputed from train+val CPU series

## Memory score

`memory_score = 1 - pct_capture_within_96` (clipped to [0, 1])

Higher score → more long-range dependence not captured within the G96 window.

## Memory classes

Tertiles of memory_score: low / medium / high.

## Analyses

- Pearson/Spearman: memory_score vs Δ MAE (G96→G288)
- Scatter plots with trend lines
- Bootstrap subgroup statistics by memory class
- Explicit answers: do high-memory workloads benefit more?

Outputs: `experiments/ggtce_*/memory_analysis/`
