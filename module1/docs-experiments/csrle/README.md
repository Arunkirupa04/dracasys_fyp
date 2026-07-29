# Controlled Synthetic Residual Learnability Experiment (CSRLE)

## Overview

**CSRLE** is a post-hoc, controlled validation experiment testing whether the **frozen Hybrid Prophet + GRU pipeline** can learn forecastable residual structure under known synthetic conditions.

| Item | Value |
|------|-------|
| **Protocol** | **v2_boundary_safe** (Amendment 1, 2026-07-25) |
| **Status** | **Stage 2 complete** (2026-07-25). B0/B1/C0/C1 trained and evaluated. |
| **Authoritative v2 run** | `experiments/synthetic_residual_learnability_2026-07-25_164307/` |
| **Stage 1 artifacts** | `.../stage1_control_reproduction/` |
| **Stage 2 artifacts** | `.../stage2_synthetic_gru/` |
| **v1 audit run (preserved)** | `experiments/synthetic_residual_learnability_2026-07-25_163013/` (failed clip gate) |
| **Evaluable containers** | 99 |
| **κ target** | 0.10 (κ_eff = α_c · 0.10) |

## Protocol history

| Version | Outcome |
|---------|---------|
| **v1** | Temporal + retention valid; **clip gate FAILED** → no synthetic GRU |
| **v2** | Boundary-safe α_c; **all pre-GRU gates PASSED** → pending GRU approval |

See [protocol_amendments.md](protocol_amendments.md) for full amendment record.

## Stage 2 — Synthetic GRU (COMPLETE)

| Item | Result |
|------|--------|
| Conditions trained | B0, B1, C0, C1 (independent GRUs) |
| Condition A | Reference only (Stage 1, not retrained) |
| Primary comparison | B0 vs C0, B1 vs C1 |
| Outcome | Modest B0/C0 and B1/C1 residual-learning gains; R² still negative; Ridge > GRU on B0 |

See [stage2_report.md](stage2_report.md) and [results.md](results.md).

## Stage 1 — Condition A Hybrid reproduction (PASS)

| Gate | Result |
|------|--------|
| Frozen eval path sanity | **PASS** (max diff ≈ 0) |
| Reproduction gate | **PASS** |
| B0/B1/C0/C1 GRU | **Trained (Stage 2)** |

Authoritative: `stage1_control_reproduction/evaluation/reproduction_summary.json`

**Recommendation:** Stage 2 complete — see [stage2_report.md](stage2_report.md) for thesis use.

## Pre-GRU v2 summary (authoritative: `pre_gru_summary.json`)

| Gate | B0 | B1 | C0 | C1 |
|------|:--:|:--:|:--:|:--:|
| Generator | PASS | PASS | PASS | PASS |
| Amplitude | PASS | PASS | — | — |
| Clip | PASS | PASS | PASS | PASS |
| Retention | PASS | PASS | — | — |
| Permitted GRU | Yes | Yes | Yes | Yes |

**Frozen config:** `experiments/synthetic_residual_learnability_2026-07-25_164307/config/synthetic_config_frozen.json`

## Documentation index

| Document | Purpose | Status |
|----------|---------|--------|
| [protocol_amendments.md](protocol_amendments.md) | Amendment chronology | Complete |
| [methodology.md](methodology.md) | Locked vs CSRLE methodology | Complete (v2) |
| [experimental_design.md](experimental_design.md) | Hypotheses, controls | Complete |
| [synthetic_generation.md](synthetic_generation.md) | Generator + α_c spec | Complete (v2) |
| [validation_protocol.md](validation_protocol.md) | All gates with v2 results | Complete |
| [prophet_retention_analysis.md](prophet_retention_analysis.md) | P_A/P_B decomposition | Complete (v2) |
| [implementation_log.md](implementation_log.md) | Build chronology | Ongoing |
| [results.md](results.md) | Evaluation results | Complete (Stage 2) |
| [stage2_report.md](stage2_report.md) | Stage 2 final report + Q1–Q10 | Complete |
| [discussion.md](discussion.md) | Interpretation | Complete |
| [reproducibility.md](reproducibility.md) | Reproduction commands | Complete (v2) |

## Notebooks (read-only visualization)

| Notebook | Purpose |
|----------|---------|
| [notebooks/csrle_visualization.ipynb](../../notebooks/csrle_visualization.ipynb) | CSRLE dashboard — α, synthetic data, metric matrices, Stage 1/2 summaries, frozen plots |
| [notebooks/csrle_ridge_vs_gru_analysis.ipynb](../../notebooks/csrle_ridge_vs_gru_analysis.ipynb) | Why Ridge beats GRU on B0 (diagnostic analysis) |

## Code (CSRLE-specific)

```
utils/csrle/boundary.py     # α_c computation (v2)
utils/csrle/                # generators, dataset, retention, gates
utils/csrle/stage2_*.py           # Stage 2 training, metrics, baselines, plots
scripts/run_csrle_pre_gru_validation.py
scripts/run_csrle_stage1_control_reproduction.py
scripts/run_csrle_stage2_synthetic_gru.py
```

## Read-only dependencies

Frozen `data/`, `utils/hybrid_*.py`, baseline experiments — unchanged.
