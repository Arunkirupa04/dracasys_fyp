# Implementation Plan

**Status: Approved in principle — protocol v1.1. No code until Phase 1.**

This document describes **what will be built** after `lfhe_config_frozen.json` is written. It contains **no code**.

---

## Phase 0 — Protocol freeze (pre-code)

| Step | Deliverable |
|------|-------------|
| 0.1 | Protocol v1.1 approved (diagnostic extensions + λ documentation) |
| 0.2 | Write `experiments/loss_function_hypothesis_<timestamp>/config/lfhe_config_frozen.json` |
| 0.3 | Record literal hyperparameters from Stage 2 B0 `training_metadata.json` into frozen config |
| 0.4 | Record frozen λ = 1.0 with justification pointer to `loss_function_selection.md` |
| 0.5 | Tag git commit: `docs: freeze LFHE protocol v1.1` (documentation only) |

**Frozen config must include:** λ (= 1.0, non-tunable), ε, ddof, loss formula string, horizon bands, diagnostic output paths, seeds, paths to CSRLE data (read-only), reproduction tolerances, success thresholds.

### λ freeze (implementation reminder)

- **Do not** implement λ as a CLI sweep or config grid.
- **Do not** select λ by validation performance.
- Single literal `lambda_disp: 1.0` in frozen JSON only.

---

## Phase 1 — Isolated code modules (post-approval)

| Module | Purpose |
|--------|---------|
| `utils/lfhe/loss.py` | DA-MSE and MSE (control) as callable Keras losses |
| `utils/lfhe/training.py` | Clone Stage 2 training loop with injectable loss |
| `utils/lfhe/evaluation.py` | Reuse CSRLE Stage 2 metric functions + diagnostic metrics |
| `utils/lfhe/diagnostics.py` | Horizon variance recovery + energy recovery (descriptive) |
| `utils/lfhe/plots.py` | Publication-quality PNG/PDF exports |
| `scripts/run_lfhe_experiment.py` | Orchestrator: MSE-B0, DA-B0, optional A arms |
| `scripts/build_lfhe_visualization_notebook.py` | Generate `notebooks/lfhe_visualization.ipynb` |

**Constraints:**

- Import from `utils/csrle/stage2_*` where possible — **do not edit** Stage 2 modules in place.
- Prophet + sequence builders read frozen CSRLE paths only.
- New artifacts under `experiments/loss_function_hypothesis_<timestamp>/` only.

---

## Phase 2 — Validation before training

| Step | Check |
|------|-------|
| 2.1 | Unit test: DA-MSE = MSE when σ̂ = σ_y |
| 2.2 | Unit test: gradient pushes σ̂ upward when under-dispersed |
| 2.3 | Unit test: one-sided penalty — over-dispersed σ̂ > σ_y → L_disp = 0 |
| 2.4 | Dry-run 1 batch forward/backward on synthetic tensor |

---

## Phase 3 — Training runs (ordered)

| Order | Run | Blocking gate |
|-------|-----|---------------|
| 1 | LFHE-MSE-B0 | Must pass reproduction vs Stage 2 B0 |
| 2 | LFHE-DA-B0 | Only after step 1 passes |
| 3 | LFHE-MSE-A (optional) | Independent |
| 4 | LFHE-DA-A (optional) | After step 3 |

**Estimated compute:** 2× CSRLE B0 GRU trains (~12–15 epochs each historically).

---

## Phase 4 — Evaluation, diagnostics, and reporting

| Output | Location |
|--------|----------|
| Per-arm `extended_metrics.csv` | `.../b0_lc_mse/evaluation/`, `.../b0_lc_da_mse/evaluation/` |
| Horizon variance recovery CSVs | `.../*/evaluation/horizon_variance_recovery_*.csv` |
| Energy recovery CSVs | `.../*/evaluation/energy_recovery_*.csv` |
| Paired bootstrap JSON (primary) | `.../comparisons/paired_mse_vs_da_mse.json` |
| Paired bootstrap JSON (diagnostic) | `.../comparisons/horizon_variance_recovery_paired_bootstrap.json`, `energy_recovery_paired_bootstrap.json` |
| Cohort summary CSV | `.../comparisons/lfhe_summary.csv` |
| Training histories | per-arm `training/training_history.csv` |
| Plots (PNG + PDF) | `.../plots/` — std ratio, r scatter, loss curves, horizon variance, energy distribution |
| Final report | `docs/loss_function_experiment/results.md` + `lfhe_final_report.json` |
| Visualization notebook | `notebooks/lfhe_visualization.ipynb` |

Reuse CSRLE Stage 2 plotting conventions (matplotlib, publication fonts per visualization skill).

### Notebook requirements (`notebooks/lfhe_visualization.ipynb`)

Dedicated sections (minimum):

1. **Configuration** — paths, arms, frozen λ
2. **Primary metrics** — MSE vs DA-MSE (r, std ratio, MAE)
3. **Horizon-wise variance recovery** — interactive line/box plots by band; cohort table
4. **Residual energy recovery** — ERR distribution; MSE vs DA-MSE scatter
5. **Horizon collapse interpretation** — descriptive pattern (monotonic vs flat across bands)
6. **MSE vs DA-MSE comparison** — overlay diagnostics for both analyses

**Figure policy:** Every figure displays **inline** in the notebook **and** exports to experiment `plots/` as **PNG and PDF**.

Optional: `SAVE_FIGS` flag in notebook (default: load from pre-exported experiment plots if training already complete).

---

## Phase 4b — Diagnostic computation (post-eval, descriptive only)

| Step | Action |
|------|--------|
| 4b.1 | Compute horizon-band std ratio per container (5 bands) for each arm |
| 4b.2 | Compute cohort mean, median, 95% CI per band |
| 4b.3 | Compute ERR per container; cohort + paired bootstrap |
| 4b.4 | Generate plots (PNG + PDF) |
| 4b.5 | **Do not** feed diagnostics into hypothesis verdict logic |

See [diagnostic_analysis.md](diagnostic_analysis.md).

---

## Phase 5 — Decision and documentation

| Step | Action |
|------|--------|
| 5.1 | Apply success_failure_criteria.md mechanically |
| 5.2 | Write `results.md` and `discussion.md` in LFHE docs |
| 5.3 | Update thesis chapter — cite LFHE as **prospective** objective test |
| 5.4 | **Do not** modify CSRLE Stage 2 conclusions — LFHE extends them |

---

## Explicit non-goals (implementation)

- No modification of `notebooks/hybrid_model.ipynb`.
- No Ridge retraining.
- **No λ grid search or λ CLI sweep.**
- No architecture search (hidden units, layers).
- No Prophet changes.
- **No use of diagnostic metrics for model selection or hypothesis verdict.**

---

## Approval checklist before Phase 1

- [x] DA-MSE formulation approved (λ = 1.0, one-sided, per-sequence std)
- [x] λ justification documented (frozen, not tuned)
- [x] Diagnostic analyses documented (descriptive only)
- [x] B0-only primary accepted
- [x] Success thresholds (std ratio ≥ 0.12, etc.) accepted — **unchanged**
- [x] Reproduction tolerances accepted
- [ ] Frozen config JSON written
- [ ] Experiment directory naming convention accepted
- [ ] Frozen CSRLE artifact immutability acknowledged

---

## Rollback plan

If LFHE invalidates control reproduction:

1. Stop DA-MSE run.
2. Diff training pipeline against Stage 2 `stage2_training.py` line-by-line.
3. Fix implementation; restart **both** arms from scratch.
4. Do not patch DA-MSE only.
