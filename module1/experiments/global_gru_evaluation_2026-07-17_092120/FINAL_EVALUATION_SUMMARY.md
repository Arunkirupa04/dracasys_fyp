# Global GRU v1 — Final Evaluation Summary

**Evaluation timestamp:** 2026-07-17T09:21:25Z (completed)  
**Verification timestamp:** 2026-07-17T09:28:10Z  
**Plot generation:** Phase 5 Task 5  
**Artifact persistence:** Phase 5 Task 6 (2026-07-17)

---

## Experiment Identifier

`global_gru_evaluation_2026-07-17_092120`

**Authoritative run directory:** `experiments/global_gru_evaluation_2026-07-17_092120/`  
**Frozen reference (official baseline evaluation):** `experiments/global_gru_reference_2026-07-17/`

---

## Cohort

| Item | Value |
|------|-------|
| Selected containers | 100 |
| Evaluated containers | **99** |
| Skipped containers | **1** |
| Failures | **0** |

### Skipped containers

| Container ID | Reason |
|--------------|--------|
| `c_14674` | missing from frozen train/val data |

### Evaluated containers

Full per-container metrics: `evaluation/evaluation_df.csv` (99 rows).

Demo container: `c_11461` — MAE 2.7000, RMSE 3.4544, MAPE 25.8092.

---

## Aggregate Metrics (Day 1, real CPU %)

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| MAE | 2.0627 | 2.4701 | 0.0029 | 13.6246 |
| RMSE | 2.7559 | 3.1039 | 0.0036 | 15.9666 |
| MAPE | 118.9320 | 659.0400 | 6.9068 | 6419.7039 |

**Source:** `evaluation/evaluation_summary.csv`

---

## Verification Status

**All checks: PASS** (2026-07-17T09:28:10Z)

| Script | Result |
|--------|--------|
| `verify_global_gru_data_split.py` | **PASS** |
| `verify_global_gru_inference.py` | **PASS** |
| `verify_global_gru_evaluation.py` | **PASS** |
| `verify_global_gru_train_eval_split.py` | **PASS** |
| `verify_global_gru_methodology_parity.py` | **PASS** |

**Log:** `verification/verification_notes.md`  
**Full stdout:** `verification/verification_run.log`

---

## Generated Plots

| Plot | Path |
|------|------|
| Demo Actual vs Predicted (`c_11461`) | `plots/actual_vs_predicted/c_11461.png` (+ `.pdf`) |
| Sample — best MAE (`c_13308`) | `plots/actual_vs_predicted/c_13308.png` (+ `.pdf`) |
| Sample — worst MAE (`c_12237`) | `plots/actual_vs_predicted/c_12237.png` (+ `.pdf`) |
| Sample — median MAE (`c_15630`) | `plots/actual_vs_predicted/c_15630.png` (+ `.pdf`) |
| Sample panel (4 containers) | `plots/actual_vs_predicted_samples.png` (+ `.pdf`) |
| Error distribution (Day-1 MAE) | `plots/error_distribution.png` (+ `.pdf`) |

**Plot metadata:** `plots/plot_metadata.json`

---

## Saved Artifact Locations

| Category | Path (under experiment directory) |
|----------|-----------------------------------|
| Per-container metrics | `evaluation/evaluation_df.csv` |
| Aggregate summary | `evaluation/evaluation_summary.csv` |
| Run metadata | `evaluation/evaluation_metadata.json` |
| Inference cache | `evaluation/inference_cache.pkl` |
| Verification record | `verification/verification_notes.md` |
| Plots | `plots/` |

---

## Frozen Reference Copy

Verified evaluation outputs copied to `experiments/global_gru_reference_2026-07-17/`:

- `evaluation/evaluation_df.csv`
- `evaluation/evaluation_summary.csv`
- `evaluation/evaluation_metadata.json`
- `plots/` (all Phase 5 figures)
- `verification/verification_notes.md` (updated with Phase 5 results)
- `config/baseline_metadata.json` (aggregate evaluation metrics)

**Immutability:** These frozen-reference evaluation files represent the official Global GRU v1 baseline evaluation. Re-runs must use a new `global_gru_evaluation_*` directory.

---

## Scope Note

This evaluation establishes evidence for the Global GRU baseline under the shared 99-container Day 1 protocol. **No Hybrid vs Global GRU performance comparison** is made in Phase 5 — comparative analysis is deferred to Phase 6.
