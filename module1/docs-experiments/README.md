# DracaSys Module 1 — Project Documentation

Documentation for **Module 1: Long-Term Trend and Seasonal Forecasting** of the Final Year Research Project *Intelligent Deployment Helper for Containerized Applications*.

These documents record a read-only audit of the repository (July 2026) and compare the current implementation against the intended research methodology.

---

## Documents

| Document | Description |
|----------|-------------|
| [Peak-Aware Hybrid Research](peak-aware-hybrid/README.md) | **Peak-Aware study** — research log by phase (baseline freeze, peak exploration, learning design, implementation, evaluation) |
| [Global GRU Baseline](global-gru-baseline/README.md) | **Global GRU study — complete** — research log by phase (planning through methodology comparison); study track closed |
| [Hybrid Prophet + GRU](hybrid-prophet-gru.md) | **Final implementation** — training, evaluation, utilities, artifacts, model-loading workflow, and reproducibility |
| [Implementation Roadmap](implementation-roadmap.md) | **Start here** — prioritized plan: fix current code → complete research → future work |
| [Project Audit](project-audit.md) | Repository structure, file inventory, end-to-end workflow, and artifact status |
| [Methodology Gap Analysis](methodology-gap-analysis.md) | Implementation vs. research design — compliance, leakage, deviations, and issues |
| [Data Pipeline Reference](data-pipeline-reference.md) | Stage-by-stage data flow from raw CSV to final prediction |
| [Recommended Evaluation Protocol](recommended-evaluation-protocol.md) | Target evaluation methodology to fix gaps before thesis-quality results |

---

## Quick Status (July 2026)

| Area | Status |
|------|--------|
| Preprocessing pipeline | Mostly aligned with research design |
| Hybrid Prophet + GRU | Refactored; modular utils; Day 1 primary eval; input window locked to 96; frozen reference |
| Global GRU baseline | **Complete** — Phases 1–6; official reference `experiments/global_gru_baseline_2026-07-17_121748/` (supersedes `global_gru_reference_2026-07-17/`) |
| Cross-architecture comparison | **Complete** — Hybrid vs Global GRU (`experiments/hybrid_vs_global_2026-07-17_095147/`); Hybrid lower mean MAE/RMSE on 99-container cohort (regenerated with corrected Global baseline, 2026-07-17) |
| Unseen container evaluation | Not implemented |
| Reproducibility infrastructure | Incomplete (no requirements.txt, seeds, experiment logs) |

---

## Related Project Standards

Project agent skills (research context and conventions) live in `.cursor/skills/`:

- `dracasys/` — research scope and Module 1 specification
- `python-coding-standards/` — modular Python conventions
- `experiments/` — reproducibility and metrics
- `visualization/` — publication-quality plots
