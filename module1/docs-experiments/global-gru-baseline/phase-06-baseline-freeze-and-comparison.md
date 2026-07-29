# Phase 6 — Baseline Freeze & Methodology Comparison

[← Overview](README.md) · [Phase 5 Evaluation](phase-05-evaluation.md)

---

## 7. Phase 6 — Baseline Freeze & Methodology Comparison

### 7.1 Scope Statement

> This phase compares two **frozen forecasting methodologies** under an identical preprocessing, training, and evaluation protocol. Conclusions apply only to the **implemented configurations** used in this study and should not be generalized to all Hybrid or GRU forecasting approaches.

### 7.2 Objective

Complete the Global GRU baseline research track by finalizing freeze governance, then producing a **defensible forecasting methodology comparison** (Hybrid Prophet + GRU vs Global GRU v1) suitable for thesis reporting — without retraining either model or modifying frozen artifacts.

Phase 6 completes Research Objective 2 (*compare Hybrid Prophet + GRU against Global GRU under fair conditions*) and establishes the permanent two-baseline handoff for future Peak-Aware Global GRU research.

**Global GRU frozen reference:** `experiments/global_gru_reference_2026-07-17/` (evaluation immutable since Phase 5 Task 6)  
**Global GRU evaluation run:** `experiments/global_gru_evaluation_2026-07-17_092120/`  
**Hybrid frozen control:** `experiments/baseline_reference_2026-07-14/`  
**Specification authority:** [Phase 0.5 — Global GRU Baseline Specification](phase-03-5-baseline-specification.md)

### 7.3 Locked Inputs (Read-Only — Do Not Recompute Metrics)

| Item | Value |
|------|-------|
| Global GRU model | `global_gru_v1` — trained Phase 4 |
| Hybrid control | Frozen baseline Hybrid Prophet + GRU |
| Cohort | 99 evaluable containers (`c_14674` skipped) |
| Horizon | Day 1 — 96 validation timesteps per container |
| Delta sign convention | **Global − Hybrid** (positive = Global higher error) |
| Peak-Aware | **Out of scope** |
| Intentional difference | **Forecasting methodology only** |

### 7.4 Comparison Philosophy

| Fixed (identical) | Differs by design |
|-------------------|-------------------|
| Preprocessing pipeline | Forecasting methodology |
| Temporal data split | |
| Evaluation procedure | |
| Metrics and reporting | |
| Container cohort | |

The research question is which **forecasting approach** performs better under the locked protocol — not which neural network configuration is optimal.

### 7.5 Phase 6 Constraints

| Constraint | Rule |
|------------|------|
| No retraining | Do not retrain Global GRU or Hybrid |
| No methodology redesign | Do not change evaluation protocol |
| Read-only baselines | Frozen references remain unmodified |
| Verification before finalize | Comparison verification must PASS before plots/discussion are locked |
| Findings vs explanations | Label claims explicitly (see §7.6) |
| Peak-Aware Global GRU | **Deferred** — separate study track |
| Unseen container eval | **Deferred** — post-freeze follow-on |

### 7.6 Writing Standard — Findings vs Possible Explanations

| Label | Definition | Example phrasing |
|-------|------------|------------------|
| **Experimental finding** | Directly supported by locked metrics | "Under the shared protocol, Global GRU mean Day 1 MAE was 2.063 compared to Hybrid's 1.746." |
| **Possible explanation** | Hypothesis not independently measured | "One possible explanation is that Hybrid's Prophet component captures daily seasonality that the Global GRU methodology must learn implicitly." |

| Avoid (overstated) | Prefer (careful) |
|--------------------|------------------|
| "The smaller GRU causes worse performance." | "Under the locked Global GRU methodology, mean Day 1 MAE was X compared to Hybrid's Y." |
| "Global GRU is inferior." | "The Global GRU forecasting methodology did / did not outperform Hybrid on overall Day 1 accuracy under identical evaluation conditions." |

### 7.7 Metric Interpretation Guidance (Task 6 Discussion Only)

This guidance affects **interpretation prose only** — it does not change the evaluation protocol or reported metrics.

| Metric | Role in discussion |
|--------|-------------------|
| **MAE** | **Primary** indicator of forecasting accuracy |
| **RMSE** | **Primary** indicator (penalizes large errors) |
| **MAPE** | **Supporting evidence only** — near-zero CPU utilization can produce disproportionately large percentage errors even with ε = 0.01 |

Rank methodologies primarily by MAE and RMSE; cite MAPE as supplementary context with appropriate caution.

---

## 7.8 Module Responsibility Split

```
Frozen evaluation_df.csv (both references)
        ↓
utils/methodology_comparison.py
        ↓
comparison_table.csv, per_container_delta.csv, phase6_comparison.json

utils/global_comparison_plotting.py
        ↓
cross-methodology figures (PNG + PDF)
```

| Module | Responsibility |
|--------|----------------|
| `utils/methodology_comparison.py` | Load frozen eval_dfs, cohort alignment, comparison table, per-container deltas, summary stats |
| `utils/global_comparison_plotting.py` | Side-by-side Actual vs Predicted, delta histogram, scatter, overlay histogram |
| `scripts/build_hybrid_vs_global_comparison.py` | Build comparison outputs (Task 3) |
| `scripts/verify_hybrid_vs_global_comparison.py` | Verify comparison integrity (Task 4) |
| `scripts/run_hybrid_vs_global_comparison_plots.py` | Generate comparison plots (Task 5) |

**Do not** place cross-methodology logic in `global_evaluation.py` or `global_plotting.py`.

---

## 7.9 Task Plan

| Task | Focus | Status |
|------|-------|--------|
| 1 | Complete freeze governance record | **Complete** |
| 2 | Implement `utils/methodology_comparison.py` | **Complete** |
| 3 | Build comparison tables and delta files | **Complete** |
| 4 | Run comparison verification suite | **Complete** |
| 5 | Generate cross-methodology comparison plots | **Complete** |
| 6 | Discussion, research answers, and `FINAL_COMPARISON_SUMMARY.md` | **Complete** |
| 7 | Lock Phase 6 record and update project docs | **Complete** |

### Workflow

Comparison build and verification remain **separate tasks**. Verification must PASS before comparison artifacts are finalized for discussion and reference updates.

```
Task 1 → Freeze governance
Task 2 → methodology_comparison.py
Task 3 → Build comparison tables and delta files
Task 4 → Run comparison verification suite
Task 5 → Generate comparison plots
Task 6 → Discussion and research answers
Task 7 → Lock Phase 6
```

One task at a time with approval between tasks — same workflow as Phases 4 and 5.

---

### Task 1 — Complete Freeze Governance Record

**Status:** Complete

Phase 4 froze implementation; Phase 5 Task 6 froze evaluation. Task 1 updated governance documents only — **no re-copy** of evaluation artifacts.

#### Updates applied

| File | Change |
|------|--------|
| `experiments/global_gru_reference_2026-07-17/FREEZE_STATEMENT.md` | Status → evaluation complete; cohort metrics; Phase 6 comparison pointer; immutability policy |
| `experiments/global_gru_reference_2026-07-17/config/baseline_metadata.json` | `methodology_comparison_status: pending_phase_6`; `hybrid_control_reference`; Hybrid summary metrics for side-by-side provenance |

#### Freeze completeness verification

| Check | Result |
|-------|--------|
| `models/`, `evaluation/`, `plots/`, `verification/`, `config/` populated | **PASS** |
| `evaluation_status == complete_phase_5` | **PASS** |
| `methodology_comparison_status == pending_phase_6` | **PASS** |
| `evaluation_df` has 99 rows; `c_14674` absent | **PASS** |
| Hybrid reference `evaluation_df.csv` exists (read-only) | **PASS** |
| `experiments/baseline_reference_2026-07-14/` git status clean | **PASS** |

Task 1 is complete. Ready for Task 2 (`utils/methodology_comparison.py`).

---

### Task 2 — Implement `utils/methodology_comparison.py`

**Status:** Complete

**File:** `utils/methodology_comparison.py`

| Function | Responsibility |
|----------|----------------|
| `load_frozen_evaluation_dfs()` | Load Hybrid + Global `evaluation_df.csv` from frozen references |
| `load_frozen_evaluation_summaries()` | Load frozen `evaluation_summary.csv` from both references |
| `assert_cohort_alignment()` | Same 99 container IDs; `c_14674` absent; no duplicates |
| `build_comparison_table()` | Per-methodology mean / std / min / max for MAE, RMSE, MAPE |
| `build_per_container_delta()` | `delta_* = global_* - hybrid_*` |
| `summarize_deltas()` | Mean delta, improved / worsened / tied counts, std |
| `build_experiment_config()` | Shared protocol snapshot |
| `build_comparison_metadata()` | Provenance record for `comparison_metadata.json` |

**Delta sign convention (locked):** `Global − Hybrid` (`global_minus_hybrid`).

#### Task 2 verification

| Check | Result |
|-------|--------|
| Import + load frozen eval_dfs | **PASS** |
| Cohort alignment (99 containers) | **PASS** |
| `comparison_table` means match frozen summaries | **PASS** |
| Delta columns = Global − Hybrid | **PASS** |
| `per_container_delta` row count | **99** |
| MAE improved / worsened (Global better / worse) | **30 / 69** |

Task 2 is complete. Ready for Task 3 (`scripts/build_hybrid_vs_global_comparison.py`).

---

### Task 3 — Build Comparison Tables and Delta Files

**Status:** Complete

**Runner:** `scripts/build_hybrid_vs_global_comparison.py`

**Experiment directory:** `experiments/hybrid_vs_global_2026-07-17_095147/`

| File | Content |
|------|---------|
| `comparison_table.csv` | Per-methodology aggregate metrics |
| `per_container_delta.csv` | 99-row paired deltas (`Global − Hybrid`) |
| `experiment_config.json` | Shared protocol parameters |
| `phase6_comparison.json` | Means, deltas, improved/worsened counts, timestamps |
| `comparison_metadata.json` | Full provenance (§7.10) |

#### Primary results (Day 1, real CPU %, 99 containers)

| Methodology | MAE (mean) | RMSE (mean) | MAPE (mean) |
|-------------|------------|-------------|-------------|
| Hybrid Prophet + GRU | 1.7459 | 2.3878 | 111.9403 |
| Global GRU v1 | 2.0627 | 2.7559 | 118.9320 |
| **Δ (Global − Hybrid)** | **+0.3168** | **+0.3681** | **+6.9917** |

#### Per-container MAE outcome

| Outcome | Count |
|---------|-------|
| Global better (lower MAE) | **30** |
| Global worse (higher MAE) | **69** |

#### Task 3 verification

| Check | Result |
|-------|--------|
| `comparison_table` has 2 methodology rows | **PASS** |
| `per_container_delta` has 99 rows | **PASS** |
| No duplicate container IDs | **PASS** |
| Delta convention Global − Hybrid | **PASS** |
| No NaN in delta outputs | **PASS** |

Task 3 is complete. Comparison outputs saved; verification is Task 4.

---

### Task 4 — Run Comparison Verification Suite

**Status:** Complete

**Script:** `scripts/verify_hybrid_vs_global_comparison.py`

**Experiment:** `experiments/hybrid_vs_global_2026-07-17_095147/`

#### Comparison integrity checks

| Check | Result |
|-------|--------|
| Both frozen datasets contain same 99 container IDs | **PASS** |
| `c_14674` absent from both datasets | **PASS** |
| No duplicate `container_id` after merge | **PASS** |
| Deltas = Global − Hybrid (MAE, RMSE, MAPE) | **PASS** |
| No NaN in comparison outputs | **PASS** |
| `comparison_table.csv` aggregates match frozen summaries | **PASS** |
| Delta summary matches `phase6_comparison.json` | **PASS** |
| Saved artifacts match frozen-source recompute | **PASS** |

#### Saved verification artifacts

| File | Location |
|------|----------|
| `verification/comparison_verification_notes.md` | Comparison experiment dir |
| `verification/comparison_verification_run.log` | Full stdout |

**Gate satisfied:** Tasks 5–7 may proceed using verified comparison artifacts.

Task 4 is complete.

---

### Task 5 — Generate Cross-Methodology Comparison Plots

**Status:** Complete

**Module:** `utils/global_comparison_plotting.py` (separate from `global_plotting.py`)

**Runner:** `scripts/run_hybrid_vs_global_comparison_plots.py`

**Experiment:** `experiments/hybrid_vs_global_2026-07-17_095147/`

| Plot | Output | Description |
|------|--------|-------------|
| Side-by-side Actual vs Predicted | `plots/side_by_side/c_11461.{png,pdf}` | Demo container — shared y-axis, both methodologies |
| Side-by-side samples | `plots/side_by_side_samples.{png,pdf}` | Best / worst / median ΔMAE containers |
| MAE delta histogram | `plots/mae_delta_histogram.{png,pdf}` | Distribution of `delta_day1_mae` (Global − Hybrid) |
| Scatter | `plots/mae_scatter.{png,pdf}` | Hybrid MAE vs Global MAE (99 points, 45° reference) |
| Error distribution overlay | `plots/mae_distribution_overlay.{png,pdf}` | Per-container MAE — both methodologies |

**Sample containers (ΔMAE = Global − Hybrid):**

| Role | Container | ΔMAE |
|------|-----------|------|
| Demo | `c_11461` | +1.2456 |
| Best (Global better) | `c_12237` | −4.0511 |
| Worst (Global worse) | `c_15640` | +4.0760 |
| Median ΔMAE | `c_15179` | +0.0768 |

**Inference arrays:**

| Source | Path | Containers |
|--------|------|------------|
| Global GRU | `experiments/global_gru_evaluation_2026-07-17_092120/evaluation/inference_cache.pkl` | Plot subset (4) |
| Hybrid | `experiments/hybrid_vs_global_2026-07-17_095147/inference_cache/hybrid_inference_cache.pkl` | Plot subset (4) — generated from frozen `models/hybrid_gru.keras` + `residual_stats.pkl` |

Hybrid inference is **visualization-only**; locked comparison metrics remain from frozen evaluation CSVs.

**Plot metadata:** `plots/plot_metadata.json`

**Result:** 10 figure files (5 categories × PNG + PDF). Plot generation: **PASS**.

Task 5 is complete.

---

### Task 6 — Discussion, Research Answers, and `FINAL_COMPARISON_SUMMARY.md`

**Status:** Complete

**Executive summary:** `experiments/hybrid_vs_global_2026-07-17_095147/FINAL_COMPARISON_SUMMARY.md`

#### Research question answers

| Question | Answer (experimental finding) |
|----------|-------------------------------|
| Overall Day-1 accuracy? | Hybrid lower mean MAE (1.746) and RMSE (2.388) than Global GRU (2.063 / 2.756) — MAE/RMSE primary |
| Which containers favour which methodology? | Global better on **30** containers; Hybrid better on **69** (MAE). Largest Global win: `c_12237` (ΔMAE −4.05). Largest Global loss: `c_15640` (ΔMAE +4.08) |
| Deployment trade-offs? | Hybrid: better cohort-average accuracy, per-container Prophet + GRU inference. Global GRU: single shared model, simpler deployment, lower average accuracy in this configuration |

#### Strengths / weaknesses (both methodologies)

Documented in `FINAL_COMPARISON_SUMMARY.md` with **findings** separated from **possible explanations** (§7.6). MAE/RMSE used as primary ranking evidence; MAPE supporting only (§7.7).

**Hybrid strengths:** lower mean MAE/RMSE; strong demo performance (`c_11461`). **Hybrid weaknesses:** 30 containers where Global wins; occasional extreme errors (`c_12237`).

**Global GRU strengths:** wins on 30/99 containers; largest improvement on high-error Hybrid case. **Global GRU weaknesses:** higher mean MAE (+0.32) and RMSE (+0.37); worse on majority of cohort.

Task 6 is complete.

---

### Task 7 — Lock Phase 6 Record

**Status:** Complete

#### Updates applied

| File | Change |
|------|--------|
| `docs/global-gru-baseline/phase-06-baseline-freeze-and-comparison.md` | All tasks marked complete; Appendix A filled; exit criteria satisfied |
| `docs/global-gru-baseline/README.md` | Phase 6 **Complete**; comparison results; study track **Closed** |
| `docs/README.md` | Global GRU study complete; cross-architecture comparison **Complete** |
| `experiments/global_gru_reference_2026-07-17/config/baseline_metadata.json` | `methodology_comparison_status: complete`; comparison provenance block |
| `experiments/global_gru_reference_2026-07-17/FREEZE_STATEMENT.md` | Phase 6 comparison pointer; verification; study track closed |

#### Task 7 verification

| Check | Result |
|-------|--------|
| `methodology_comparison_status == complete` | **PASS** |
| `docs/global-gru-baseline/README.md` reflects Phase 6 complete | **PASS** |
| `docs/README.md` reflects study track closed | **PASS** |
| `FREEZE_STATEMENT.md` includes comparison experiment pointer | **PASS** |
| Frozen Hybrid reference unchanged | **PASS** |
| Peak-Aware Global GRU not started | **PASS** |

Task 7 is complete. **Phase 6 is complete (7/7 tasks).**

---

## 7.10 `comparison_metadata.json` Schema (Provenance)

| Field | Description |
|-------|-------------|
| `comparison_timestamp` | ISO UTC timestamp of comparison build |
| `experiment_identifier` | Directory name, e.g. `hybrid_vs_global_2026-07-17_HHMMSS` |
| `hybrid_frozen_reference` | Path to `experiments/baseline_reference_2026-07-14/` |
| `global_gru_frozen_reference` | Path to `experiments/global_gru_reference_2026-07-17/` |
| `hybrid_evaluation_source` | Path to Hybrid `evaluation/evaluation_df.csv` used |
| `global_evaluation_source` | Path to Global `evaluation/evaluation_df.csv` + Phase 5 run dir |
| `comparison_script` | `scripts/build_hybrid_vs_global_comparison.py` |
| `comparison_script_version` | Git commit or phase lock date |
| `verification_script` | `scripts/verify_hybrid_vs_global_comparison.py` |
| `delta_sign_convention` | `"global_minus_hybrid"` |
| `protocol_summary` | Input window, horizon, cohort, MAPE ε, metric unit |
| `evaluated_containers` | 99 |
| `skipped_containers` | `[{"container_id": "c_14674", "reason": "..."}]` |
| `outputs` | Relative paths to comparison_table, per_container_delta, phase6_comparison |

---

## 7.11 Research Questions (Module 1)

| Question | Answered by |
|----------|-------------|
| How does Global GRU compare to Hybrid on overall Day 1 accuracy? | Task 3 + Task 6 (MAE/RMSE primary) |
| Which methodology performs better on which containers? | Task 3 deltas + Task 4 verification |
| What are deployment trade-offs? | Task 6 strengths/weaknesses |

**Not answered in Phase 6 (deferred):** unseen containers, peak-period accuracy, ablations.

---

## 7.12 Future Work (After Phase 6)

| Item | Note |
|------|------|
| Peak-Aware Global GRU | Requires frozen `global_gru_reference_*` as control |
| Unseen container holdout | New experiment directory |
| Model-structure / feature ablations | Separate experiments |
| Hyperparameter optimisation | Post-freeze only |
| Day 2 recursive Global GRU | Supplementary horizon |

---

## 7.13 Exit Criteria (Phase 6 Complete)

- [x] Freeze governance record complete (Task 1)
- [x] `utils/methodology_comparison.py` implemented (Task 2)
- [x] `comparison_table.csv` and `per_container_delta.csv` saved (Task 3)
- [x] `comparison_metadata.json` includes full provenance (§7.10)
- [x] `verify_hybrid_vs_global_comparison.py` PASS (Task 4)
- [x] Comparison plots exported PNG + PDF (Task 5)
- [x] `FINAL_COMPARISON_SUMMARY.md` written (Task 6)
- [x] Strengths/weaknesses for both methodologies (Task 6)
- [x] `docs/README.md` and `docs/global-gru-baseline/README.md` updated (Task 7)
- [x] Frozen Hybrid reference unchanged
- [x] Peak-Aware Global GRU **not** started

---

## 7.14 Handoff to Future Research

| Forecasting methodology | Frozen reference |
|------------------------|------------------|
| Hybrid Prophet + GRU | `experiments/baseline_reference_2026-07-14/` |
| Global GRU v1 | `experiments/global_gru_reference_2026-07-17/` |

Peak-Aware Global GRU must treat `global_gru_reference_*` as permanent control, implement only in parallel modules, and never overwrite frozen reference folders.

---

## 7.15 Conclusion

Phase 6 is **complete** (7/7 tasks). The Global GRU baseline research track is **closed**.

Under the locked 99-container Day-1 protocol, Hybrid Prophet + GRU achieved lower mean Day-1 MAE (1.746) and RMSE (2.388) than Global GRU v1 (2.063 / 2.756), while Global GRU was more accurate on 30 of 99 containers. Verified comparison artifacts, plots, and executive summary are preserved under `experiments/hybrid_vs_global_2026-07-17_095147/`.

**Handoff:** Both frozen references (`baseline_reference_2026-07-14/` and `global_gru_reference_2026-07-17/`) are permanent controls for future Peak-Aware Global GRU and related studies.

---

## Appendix A — Comparison Results Template

### Primary Results (from frozen references — Task 3 recomputes)

| Forecasting methodology | MAE | RMSE | MAPE | N |
|------------------------|-----|------|------|---|
| Hybrid Prophet + GRU | 1.746 | 2.388 | 111.94 | 99 |
| Global GRU v1 | 2.063 | 2.756 | 118.93 | 99 |
| **Δ (Global − Hybrid)** | +0.317 | +0.368 | +6.99 | — |

### Per-Container Delta Summary (Task 3–4 verified)

| Statistic | MAE | RMSE | MAPE |
|-----------|-----|------|------|
| Mean delta (Global − Hybrid) | +0.3168 | +0.3681 | +6.9917 |
| Improved / Worsened | 30 / 69 | 21 / 78 | 26 / 73 |
| Std of delta | 0.9376 | 1.1389 | 77.7206 |
| Min delta | −4.0511 (`c_12237`) | −6.8614 (`c_12237`) | −334.1769 (`c_10306`) |
| Max delta | +4.0760 (`c_15640`) | +4.3840 (`c_12231`) | +484.1570 (`c_12060`) |

---

## Appendix B — Related Documentation

| Document | Purpose |
|----------|---------|
| [Phase 0.5 Baseline Specification](phase-03-5-baseline-specification.md) | Locked research contract |
| [Phase 5 Evaluation](phase-05-evaluation.md) | Global GRU cohort evidence |
| [Hybrid Prophet + GRU](../hybrid-prophet-gru.md) | Hybrid methodology specification |
| [Peak-Aware Hybrid](../peak-aware-hybrid/README.md) | Separate Hybrid enhancement study |
| [Recommended Evaluation Protocol](../recommended-evaluation-protocol.md) | Shared protocol authority |
