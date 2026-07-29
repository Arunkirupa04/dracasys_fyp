# CSRLE Ridge vs GRU Analysis

**Status:** Read-only post-hoc explanatory analysis (2026-07-26)

## Purpose

Explain why **Ridge regression outperformed the frozen Hybrid GRU** on the CSRLE **B0 positive-control** residual-learning task, without retraining models or modifying frozen CSRLE artifacts.

## Research question

See [research_question.md](research_question.md).

## Artifacts

| Location | Role |
|----------|------|
| `experiments/csrle_ridge_vs_gru_analysis_2026-07-26_013200/` | Analysis outputs (plots, CSVs, JSON) |
| `notebooks/csrle_ridge_vs_gru_analysis.ipynb` | Interactive section-by-section notebook |
| Frozen CSRLE Stage 2 B0 | `experiments/synthetic_residual_learnability_2026-07-25_164307/stage2_synthetic_gru/b0_lc/` |

## Documentation index

| File | Content |
|------|---------|
| [methodology.md](methodology.md) | What was measured and how |
| [analysis_protocol.md](analysis_protocol.md) | Ten analysis modules |
| [results.md](results.md) | Quantitative findings |
| [discussion.md](discussion.md) | Interpretation |
| [summary.md](summary.md) | Thesis-ready answers (7 questions) |
| [limitations.md](limitations.md) | Scope boundaries |
| [reproducibility.md](reproducibility.md) | Commands |
| [implementation_log.md](implementation_log.md) | Chronology |

## Read-only guarantee

No files under frozen experiments, `data/`, `models/`, or prior CSRLE stages are modified.
