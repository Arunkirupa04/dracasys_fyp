# Phase 3 Design Lock — Peak-Aware Hybrid Prophet + GRU

**Date:** 2026-07-14  
**Status:** Complete — locked for Phase 4 implementation

## Locked Design

| Element | Choice |
|---------|--------|
| Mechanism | Timestep-weighted MSE |
| Peak definition | P90 per-container, train-only, real CPU % |
| Peak weight λ | 5 |
| Early stopping | Unweighted `val_loss`, patience 10 |
| Experimental variable | GRU training loss only |

## Formal Training Rule

> During GRU training, each peak timestep in the 96-step forecast horizon receives loss weight **λ = 5**; all other horizon timesteps receive weight **1.0**. Peak if `cpu_real >= P90(container_train)`.

## Implementation (Phase 4)

New files only:

- `utils/peak_config.py`
- `utils/peak_detection.py`
- `utils/hybrid_training_peak_aware.py`
- `scripts/run_peak_aware_hybrid_experiment.py`
- `scripts/verify_peak_aware_fairness.py`

Frozen baseline: `experiments/baseline_reference_2026-07-14/`

## References

- Full design log: `docs/peak-aware-hybrid/phase-03-learning-design.md`
- Peak definition: `experiments/peak_exploration_2026-07-14/peak_definition_decision.json`
- Machine-readable lock: `phase3_design.json`
