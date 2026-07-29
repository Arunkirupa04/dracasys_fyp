# Implementation Plan

> **Not executed in protocol stage.** Checklist for post-approval implementation.

## Phase 0 — Protocol approval

- [ ] Review `docs/global_window_experiment/` with supervisor
- [ ] Confirm variant set: G96, G192, G288 (exclude G384)
- [ ] Confirm subgroup analysis as co-primary objective

## Phase 1 — Isolated module (future)

```
utils/ggtce/
  config.py           # GGTCE_VARIANTS, frozen hyperparameters reference
  training.py         # wrap train_global_gru with input_window param (no hardcoded 96 check)
  evaluation.py       # wrap global evaluation + TMA merge
  ablation.py         # paired tests, subgroup analysis
  plots.py
  report.py

scripts/run_ggtce_experiment.py
scripts/build_ggtce_notebook.py
```

**Critical:** Remove or parameterise `EXPECTED_N_SEQUENCES` assertion in training wrapper — counts vary by window.

## Phase 2 — Experiment execution

1. Create `experiments/ggtce_{timestamp}/`
2. Write `config/ggtce_config_frozen.json`
3. Train G96 → verify against `global_gru_baseline_2026-07-17_121748`
4. Train G192, G288 independently
5. Evaluate all variants (Day-1 protocol)
6. Load TMA container table; assign memory tertiles
7. Run ablation + subgroup statistics
8. Generate plots (PNG/PDF) and `final_report.md/json`

## Phase 3 — Documentation update

- [ ] Populate `results.md` in this folder from authoritative run
- [ ] Update `summary.md` with measured outcomes
- [ ] Add thesis paragraph referencing GGTCE verdict

## Phase 4 — Notebook

- [ ] Execute `notebooks/ggtce_ablation.ipynb` (results notebook — separate from this design notebook)

## Estimated timeline

| Phase | Duration |
|-------|----------|
| Implementation | 1–2 days |
| Training (3 variants) | 2–4 hours GPU |
| Evaluation + stats | 2–3 hours |
| Documentation | 1 day |

## Dependencies

- TensorFlow environment (`tf_metal_env`)
- Frozen data artifacts (unchanged)
- TMA container-level CSV (read-only import)

## Out of scope for implementation v1

- G384 primary arm
- Architecture changes (deeper GRU, attention)
- Peak-aware loss
- Hyperparameter search
