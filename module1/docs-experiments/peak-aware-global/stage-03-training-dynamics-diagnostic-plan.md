# Validation Study VS-1 — Training Dynamics (Revised Plan)

[← Stage 3 Evaluation Plan](stage-03-evaluation-plan.md) · [Study Overview](README.md)

**Status:** Approved (with refinements below) — **not implemented**  
**Study label:** Validation Study – Training Dynamics  
**Primary experiment (unchanged):** `experiments/peak_aware_global_2026-07-17_164737/`

---

## 1. Objective

Answer **one question only:**

> Was the poor Peak-Aware Global performance primarily caused by premature early stopping?

This is **not**:

- A new methodology study
- λ tuning
- A hyperparameter search
- A replacement for the locked primary experiment

It investigates **only** the effect of EarlyStopping patience: **3 → 10**.

---

## 2. Invariants (Identical to Locked Primary)

Everything below matches the locked Peak-Aware Global experiment (`_164737`). **No changes.**

| Component | Locked value |
|-----------|--------------|
| λ | 5 |
| Peak definition | P90, train-only, per-container |
| Architecture | `build_global_gru_model()` (read-only) |
| Optimizer | Adam (default learning rate) |
| Batch size | 256 |
| Shuffle | False |
| Train/validation split | Sequence-level 80/20 |
| Selected containers | `data/selected_containers.npy` |
| Weight generation | Same `peak_thresholds.pkl` / `W_all` logic |
| Weighted `train_step` | Unchanged |
| Evaluation protocol | Frozen `evaluate_selected_containers()` |
| Random seed | 42 |
| Epochs max | 50 |
| Early stopping monitor | `val_loss` (unweighted) |
| Restore best weights | True |

**Single allowed change:**

| Setting | Primary (locked) | VS-1 validation |
|---------|------------------|-----------------|
| EarlyStopping patience | **3** | **10** |

---

## 3. Validation Experiment Design

### One run only

| Item | Value |
|------|-------|
| Label | `patience_10` |
| EarlyStopping patience | **10** |
| All other settings | Identical to primary |

**Removed from prior draft (do not run):**

- ~~Arm A (locked replay)~~
- ~~Arm C (50 epochs without EarlyStopping)~~
- ~~Optional baseline replay~~
- ~~Eval-at-checkpoint (epochs 4, 9, etc.)~~

The **final best model** selected by EarlyStopping (with `restore_best_weights=True`) is the only model evaluated.

---

## 4. Artifacts

Save under a separate validation experiment directory (§7):

| Artifact | Content |
|----------|---------|
| `training/training_history.json` | Per-epoch `loss`, `val_loss` (and `mae` / `val_mae` if emitted by Keras) |
| `training/training_metadata.json` | Required epoch fields (see §4.1) + hyperparams, patience = 10, timestamps |
| `models/global_gru_peak_aware.keras` | Best weights from EarlyStopping |
| `evaluation/evaluation_summary.csv` | Day-1 cohort aggregate (MAE, RMSE, MAPE) |
| `evaluation/evaluation_df.csv` | Per-container Day-1 metrics |
| `plots/learning_curves.png` | Train loss + validation loss vs epoch |

**Not collected:**

- Multiple intermediate checkpoints
- Per-checkpoint Day-1 evaluations
- Multi-arm summary tables beyond the primary vs VS-1 comparison

Peak thresholds and weight matrix: reuse primary run artifacts (copy/symlink `peak/` from `_164737`) — no refit.

### 4.1 Required fields in `training_metadata.json`

These fields **must** be recorded so the exact timing of the best model can be determined:

| Field | Definition |
|-------|------------|
| `epochs_run` | Total number of epochs completed before training ended (1-indexed count) |
| `best_epoch` | Epoch index (1-indexed) at which the lowest unweighted `val_loss` occurred |
| `stopped_epoch` | Epoch index (1-indexed) at which training terminated (may equal `epochs_run`) |
| `best_val_loss` | Minimum unweighted validation loss achieved at `best_epoch` |

**Note:** With `restore_best_weights=True`, the saved model corresponds to `best_epoch`, not necessarily `stopped_epoch`.

---

## 5. Comparison

Compare VS-1 (`patience_10`) against:

### A. Locked primary Peak-Aware Global (`_164737`, patience = 3)

| Metric | Source |
|--------|--------|
| Epochs run | `training_metadata.json` → `epochs_run` |
| Best epoch | `training_metadata.json` → `best_epoch` |
| Stopped epoch | `training_metadata.json` → `stopped_epoch` |
| Best validation loss | `training_metadata.json` → `best_val_loss` |
| Final Day-1 MAE | `evaluation/evaluation_summary.csv` |
| Final Day-1 RMSE | same |
| Final Day-1 MAPE | same |

### B. Frozen Global baseline (reference context only)

| Metric | Source |
|--------|--------|
| Epochs run | `global_gru_reference/.../training_metadata.json` (9) |
| Best / final val loss | same |
| Day-1 MAE / RMSE / MAPE | `global_gru_reference/evaluation/evaluation_summary.csv` |

Learning curves (VS-1 vs primary if primary history is recovered later) and the five metrics above support interpretation.

---

## 6. Decision Rule

Interpret from **observed learning curves and evaluation metrics**. No fixed percentage thresholds.

### Conclusion 1 — Early stopping is **not** the primary cause

If `patience=10` still produces **substantially worse** validation loss and Day-1 evaluation metrics than the frozen Global baseline, premature stopping is unlikely to explain the degradation. The weighted objective on direct CPU prediction is the more plausible driver.

### Conclusion 2 — Premature stopping is a **plausible contributing factor**

If `patience=10` produces a **substantial improvement** over the primary Peak-Aware Global run (patience = 3) on validation loss and/or Day-1 metrics, the primary experiment may have been stopped before the weighted objective had sufficient time to converge.

### Nuance

- Improvement over primary but not near baseline → early stopping **contributed** but does not fully explain the gap.
- Use learning curves: still improving at primary stop epoch vs plateaued under both patiences.

---

## 7. Separation From Primary Experiment

### Directory and labelling

| Item | Primary (locked) | VS-1 validation |
|------|------------------|-------------------|
| Root | `experiments/peak_aware_global_2026-07-17_164737/` | `experiments/peak_aware_global_validation_vs1_<timestamp>/` |
| Report title | Primary Peak-Aware Global experiment | **Validation Study – Training Dynamics** |
| Stage 2 design lock | **Unchanged** | Does not amend `design_lock.json` |
| Stage 3 primary eval | **Uses only `_164737`** | VS-1 excluded from primary comparison tables |
| Stage 3 methodology | **Unchanged** | VS-1 does not alter evaluation protocol |

### Reporting rules

1. Primary `_164737` remains the **official** Peak-Aware Global result under the locked design (patience = 3).
2. VS-1 is reported as supplementary evidence on the early-stopping hypothesis only.
3. VS-1 does **not** replace, supersede, or retroactively modify the Stage 2 lock or Stage 3 evaluation plan.
4. Wording: “validation study,” not “corrected experiment” or “final treatment model.”
5. **Optimization vs methodology interpretation:** If the VS-1 model materially outperforms the primary Peak-Aware Global model, the original Stage 2 training protocol will be considered **optimization-limited** rather than **methodology-limited**. Any decision to replace or retain the primary experiment will be made **only after reviewing the VS-1 evidence**.

### Machine-readable metadata

`validation_study_metadata.json` at VS-1 root:

```json
{
  "study_type": "VS-1-training-dynamics",
  "study_label": "Validation Study – Training Dynamics",
  "question": "Was poor performance primarily caused by premature early stopping?",
  "single_change": { "early_stopping_patience": { "primary": 3, "validation": 10 } },
  "primary_experiment": "experiments/peak_aware_global_2026-07-17_164737",
  "control_reference": "experiments/global_gru_reference_2026-07-17"
}
```

---

## 8. Execution Outline (After Approval)

| Step | Action |
|------|--------|
| 1 | Create VS-1 workspace + `validation_study_metadata.json` |
| 2 | Train once with patience = 10; log `training_history.json` |
| 3 | Save best model + `training_metadata.json` (including §4.1 epoch fields) |
| 4 | Run standard Day-1 evaluation on best model (same protocol as Task 2) |
| 5 | Plot learning curves (train + val loss) |
| 6 | Write comparison vs primary + baseline; record conclusion (§6) |

**Do not** modify frozen baseline code, primary artifacts, or design lock.

---

## 9. Out of Scope

- λ, P90, architecture, optimizer, LR, batch, data, or split changes
- Additional patience values or disabling EarlyStopping
- Replacing the primary model in Stage 3
- Hyperparameter search or methodology exploration

---

## 10. Approval Checklist

- [x] Single-arm design (patience 10 only) approved
- [x] Artifact list (§4) approved
- [x] Primary `_164737` remains official regardless of VS-1 outcome
- [x] Epoch timing fields (§4.1) specified
- [x] Optimization vs methodology reporting statement (§7) included
- [ ] Proceed to implementation

---

**Next step after implementation approval:** Run VS-1 workspace + one training run (patience = 10).

*Plan approved — implementation not started*
