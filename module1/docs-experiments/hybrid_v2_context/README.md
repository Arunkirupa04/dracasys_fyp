# HCERL — Hybrid v2 Context-Enriched Residual Learning

**Protocol:** `hcerl_v1.0`  
**Status:** Controlled ablation experiment (isolated from baseline)

## Research question

> Which contextual information actually improves Hybrid residual learning beyond the baseline residual-only input?

## Variants

| ID | Shape | Cumulative features |
|----|-------|---------------------|
| V0 | 96×1 | `residual_scaled` |
| V1 | 96×2 | + `prophet_yhat_scaled` |
| V2 | 96×3 | + `res_roll_std_12` |
| V3 | 96×4 | + `cpu_std` |
| V4 | 96×5 | + `is_peak_p90` |

## Code

| Component | Path |
|-----------|------|
| Module | `utils/hcerl/` |
| Runner | `scripts/run_hcerl_experiment.py` |
| Notebook | `notebooks/hybrid_v2_context_ablation.ipynb` |

## Documents

- [research_question.md](research_question.md)
- [methodology.md](methodology.md)
- [feature_engineering.md](feature_engineering.md)
- [ablation_protocol.md](ablation_protocol.md)
- [implementation_log.md](implementation_log.md)
- [results.md](results.md) — populated after run
- [discussion.md](discussion.md)
- [limitations.md](limitations.md)
- [future_work.md](future_work.md)
- [summary.md](summary.md)

## Isolation policy

Does **not** modify baseline Hybrid, Global GRU, CSRLE, LFHE, TMA, or context feasibility artifacts.
