# Global GRU Temporal Context Experiment (GGTCE)

**Protocol stage only** — no implementation, no training, no code changes.

**Version:** `ggtce_v1.0` (draft for review)  
**Date:** 2026-07-26

## Purpose

Investigate whether **increasing Global GRU input history** (96 → 192 → 288) improves Day-1 CPU forecasting, motivated by Temporal Memory Analysis (TMA) showing substantial long-range dependence in **raw CPU** that the current 96-step window does not fully capture.

## Scope boundary

| In scope | Out of scope |
|----------|--------------|
| Global GRU input window ablation | Hybrid Prophet+GRU changes |
| G96 / G192 / G288 variants | HCERL-style context features |
| TMA-linked subgroup analysis | Peak-aware training (separate track) |
| Frozen everything except `input_window` | Hyperparameter tuning |

## Documents

| File | Content |
|------|---------|
| [research_question.md](research_question.md) | Primary and secondary questions |
| [background.md](background.md) | Motivation from TMA and research chain |
| [methodology.md](methodology.md) | Frozen Global GRU methodology summary |
| [experimental_design.md](experimental_design.md) | Variant definitions and isolation rules |
| [window_analysis.md](window_analysis.md) | Sequence counts, cost, benefit estimates |
| [evaluation_protocol.md](evaluation_protocol.md) | Metrics, statistics, TMA linkage |
| [hypotheses.md](hypotheses.md) | H0, H1, alternative outcomes |
| [success_criteria.md](success_criteria.md) | Practical and statistical thresholds |
| [risk_assessment.md](risk_assessment.md) | Risks and mitigations |
| [implementation_plan.md](implementation_plan.md) | Future implementation checklist (not executed) |
| [future_work.md](future_work.md) | Post-GGTCE directions |
| [summary.md](summary.md) | Executive summary + explicit answers |

## Notebook

[`notebooks/global_window_experiment_design.ipynb`](../../notebooks/global_window_experiment_design.ipynb) — design tables, TMA summary, timeline (markdown only).

## Prior work referenced (read-only)

- Global baseline: `experiments/global_gru_baseline_2026-07-17_121748/`
- TMA: `experiments/temporal_memory_analysis_2026-07-26_170052/`
- HCERL verdict: context features did not help Hybrid; Global path remains open per TMA

## Recommendation preview

**Proceed** with G96 (control) + **G192** + **G288**. **Exclude G384** from primary arms. **Include TMA-based subgroup analysis** as a co-primary analytic objective.

See [summary.md](summary.md) for explicit answers.
