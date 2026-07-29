# Context Feature Feasibility Study (Hybrid v2 Design)

**Status:** Design and feasibility only — no model training, no new experiments, no synthetic data.

**Date:** 2026-07-26  
**Scope:** Determine whether a **context-enriched Hybrid Prophet + GRU** (Hybrid v2) is scientifically justified using the existing Alibaba dataset and all prior research findings.

## Research question

> Should the Hybrid GRU receive additional contextual features beyond the 96-step scaled Prophet residual history, and does our dataset contain such information without leakage?

## Current Hybrid input

```
CPU → Prophet → residual → GRU (96 × 1: residual_scaled only) → residual forecast → final CPU
```

## Documents

| File | Purpose |
|------|---------|
| [summary.md](summary.md) | Executive summary and explicit answers to the four feasibility questions |
| [dataset_review.md](dataset_review.md) | Task 1 — every Alibaba column, meaning, inference availability, current usage |
| [candidate_features.md](candidate_features.md) | Task 2 — full catalogue of derivable contextual features |
| [feature_analysis.md](feature_analysis.md) | Tasks 3–6 — leakage, Prophet redundancy, correlation, compact set |
| [feature_ranking.md](feature_ranking.md) | Task 10 — ranked candidate features |
| [hybrid_v2_design.md](hybrid_v2_design.md) | Task 7 — proposed input tensor layout |
| [risks.md](risks.md) | Task 9 — risks and mitigations |
| [recommendation.md](recommendation.md) | Final recommendation and decision criteria |
| [future_work.md](future_work.md) | Next steps if Hybrid v2 is pursued |

## Notebook

[`notebooks/context_feature_feasibility.ipynb`](../../notebooks/context_feature_feasibility.ipynb) — read-only dataset inspection, feature derivation examples, correlation analysis, visualizations, and conceptual conclusions. **No training.**

## Supporting artifacts (read-only analysis cache)

- `figures/` — correlation bar charts, ACF comparison, inter-feature heatmap
- `_analysis_cache/` — CSV summaries from read-only correlation scripts (not an experiment run)

## Prior research referenced

| Experiment | Relevance |
|------------|-----------|
| Residual Pattern Analysis (RPA) | Prophet residuals retain temporal structure |
| CSRLE | GRU can learn structured synthetic residuals |
| LFHE | MSE drives variance collapse; fixing dispersion ≠ fixing correlation |
| Temporal Memory Analysis (TMA) | Prophet removes long CPU memory; 96-step residual window supported |
| Ridge Residual Audit / RLLA | Real residuals weakly linearly predictable; structure persists but is hard to exploit |

## Verdict (preview)

**Conditionally justified** to investigate a **compact** context-enriched Hybrid v2 — primarily **level/regime** and **local volatility** features — not a large feature expansion. Calendar features are likely redundant post-Prophet. Multivariate Alibaba telemetry (mem, net, disk) is effectively unavailable.

See [recommendation.md](recommendation.md) for the full ranked feature set and go/no-go criteria.
