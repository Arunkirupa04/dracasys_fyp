# Phase 3 — Peak-Aware Learning Design (Complete)

[← Overview](README.md)

---

## 4. Phase 3 — Peak-Aware Learning Design

### 4.1 Objective

Decide **how** to introduce peak-aware learning into the frozen Hybrid Prophet + GRU model, given the **locked peak definition from Phase 2**:

> A timestep is a peak if `cpu_real >= P90(container_train)` on real CPU %, per-container, train-fitted thresholds only.

This phase is **design only**: no implementation, no training, no baseline modification. Phase 3 must produce one clear, defensible design where the **only experimental change** versus the frozen baseline is the GRU **training objective**.

### 4.2 Inputs from Phase 2 (Locked)

| Item | Decision |
|------|----------|
| Peak definition | Per-container **P90**, train-only, real CPU % |
| Weighting granularity | **Timestep-level** (locked in Task 2) |
| Early stopping | **Unweighted `val_loss`** (locked in Task 4) |
| Baseline reference | `experiments/baseline_reference_2026-07-14/` |
| Peak weight λ | **λ = 5** (locked in Task 3) |

### 4.3 Phase 3 Constraints

| Constraint | Rule |
|------------|------|
| Baseline code | Do **not** modify `utils/hybrid_*.py`, `utils/sequence_utils.py`, notebooks, or `models/` |
| Preprocessing | Do **not** change frozen `data/` artifacts or preprocessing pipeline |
| Prophet | Do **not** change Prophet configuration or residual generation logic |
| Architecture | Do **not** change GRU architecture, optimizer, epochs, batch size, or shuffle |
| Peak definition | Do **not** revisit P85/P95 or add an absolute CPU floor |
| Alternative mechanisms | Do **not** implement focal loss, oversampling, quantile loss, or architecture changes in this study |
| Implementation | Deferred to Phase 4; this phase documents design only |

### 4.4 Task Plan

| Task | Focus | Status |
|------|-------|--------|
| 1 | Design scope and fairness contract | **Complete** |
| 2 | Confirm primary learning mechanism | **Complete** |
| 3 | Peak weight (λ) decision | **Complete** |
| 4 | Training protocol specification | **Complete** |
| 5 | Module and artifact plan | **Complete** |
| 6 | Lock Phase 3 design document | **Complete** |

---

### Task 1 — Design Scope and Fairness Contract

**Status:** Complete  
**Date:** 2026-07-14

#### Objective

Write the formal comparison contract between the frozen baseline Hybrid Prophet + GRU model and the proposed Peak-Aware variant **before** selecting technical implementation details. The contract must make the controlled experiment auditable: one intentional change, everything else held constant.

#### Methodology

1. Reviewed frozen baseline specification (`docs/hybrid-prophet-gru.md`, `experiments/baseline_reference_2026-07-14/config/baseline_metadata.json`).
2. Reviewed locked peak definition (`experiments/peak_exploration_2026-07-14/peak_definition_decision.json`).
3. Mapped each pipeline stage to **identical**, **intentionally different**, or **out of scope** for the Peak-Aware experiment.
4. Documented the fairness contract as a formal comparison table and checklist for Phase 4 verification.

No code, training, or data analysis was performed in this task.

#### Assumptions

- The frozen baseline in `experiments/baseline_reference_2026-07-14/` is the permanent control for all Peak-Aware comparisons.
- A scientifically fair comparison requires holding constant every factor except the GRU training loss function.
- Peak labels used during training weighting will use the **same P90 rule** locked in Phase 2; peak labels used in Phase 5 evaluation will use the same rule but are a **reporting** concern, not a model change.
- Inference and evaluation pipelines remain shared; Peak-Aware and baseline models are evaluated through the **same** `hybrid_inference.py` and `hybrid_evaluation.py` paths.

#### The Controlled Experiment

| Property | Specification |
|----------|---------------|
| **Control model** | Frozen baseline Hybrid Prophet + GRU (`hybrid_prophet_gru_v1`) |
| **Treatment model** | Peak-Aware Hybrid Prophet + GRU (parallel implementation) |
| **Single experimental variable** | GRU training loss: uniform MSE → timestep-weighted MSE |
| **Comparison type** | Paired comparison on identical data, containers, windows, and evaluation protocol |
| **Baseline metrics (Day 1)** | MAE 1.746, RMSE 2.388, MAPE 111.94 (99 containers) — from `baseline_reference_2026-07-14` |

#### Fairness Contract — Component Comparison

| Component | Baseline | Peak-Aware | Status |
|-----------|----------|------------|--------|
| **Preprocessing pipeline** | Frozen `data/` artifacts | Same frozen artifacts | **Identical** |
| **Train / validation split** | Chronological holdout from preprocessing | Same split | **Identical** |
| **Container cohort** | 100 selected; 99 evaluable (`c_14674` skipped) | Same cohort | **Identical** |
| **CPU scaling** | Per-container MinMax on `cpu_scaled` | Same scalers | **Identical** |
| **Prophet configuration** | `daily_seasonality=True`, `weekly_seasonality=False` | Same | **Identical** |
| **Prophet fitting** | Per-container refit on train period | Same | **Identical** |
| **Residual computation** | `residual = cpu_scaled - prophet_pred` | Same formula | **Identical** |
| **`res_mean`, `res_std`** | Global mean/std over all train residuals | Same computation on same residuals | **Identical** |
| **Residual target** | `residual_scaled` | Same | **Identical** |
| **Input features** | `["residual_scaled"]` only | Same | **Identical** |
| **Sequence construction** | `create_residual_sequences()`, `input_window=96`, `forecast_horizon=96` | Same windows, same ordering | **Identical** |
| **Sequence pool** | Pooled across all containers; chronological per container | Same pool | **Identical** |
| **Sequence train/val split** | First 80% / last 20% of pooled sequences (early stopping only) | Same split index | **Identical** |
| **GRU architecture** | GRU(256)→GRU(128)→GRU(64)→Dense(128,relu)→Dense(96); dropout 0.2 | Same | **Identical** |
| **Optimizer** | Adam | Same | **Identical** |
| **Training epochs (max)** | 100 | Same | **Identical** |
| **Batch size** | 64 | Same | **Identical** |
| **Shuffle** | `False` | Same | **Identical** |
| **Training metric (compiled)** | MAE (monitoring only) | Same | **Identical** |
| **Early stopping monitor** | Unweighted `val_loss` (MSE) | **Same — unweighted** | **Identical** |
| **Early stopping patience** | 10 | Same | **Identical** |
| **Restore best weights** | `True` | Same | **Identical** |
| **Random seed** | 42 (per baseline metadata) | Same seed | **Identical** |
| **GRU training loss** | Uniform MSE over 96 horizon steps | **Timestep-weighted MSE** | **Intentional change** |
| **Peak definition (training)** | Not used | P90 per-container, train-only, real CPU % | **New (weighting only)** |
| **Peak weight λ** | N/A (all weights = 1) | Fixed λ on peak timesteps (Task 3) | **New (weighting only)** |
| **Prophet at inference** | Refit per container from train data | Same | **Identical** |
| **GRU at inference** | Predict `residual_scaled` from saved weights | Same inference path | **Identical** |
| **Forecast combination** | `final_scaled = prophet_pred + residual_pred` | Same | **Identical** |
| **Inverse transform** | Per-container MinMax → real CPU % | Same | **Identical** |
| **Evaluation horizon** | Day 1 — 96 validation timesteps per container | Same | **Identical** |
| **Evaluation metrics** | MAE, RMSE, MAPE (real CPU %) | Same overall metrics | **Identical** |
| **Peak-subset evaluation** | Not reported for baseline | Phase 5 addition using same P90 rule | **Reporting only** (Phase 5) |

#### What Changes — Formal Statement

The Peak-Aware Hybrid model differs from the frozen baseline in **exactly one training-stage behaviour**:

> During GRU `model.fit`, the per-timestep squared error in the 96-step forecast horizon is multiplied by a weight `w_t`, where `w_t = λ` if the corresponding future timestep is a peak under the locked P90 rule, and `w_t = 1` otherwise. All other pipeline stages remain unchanged.

Peak awareness is a **training-time objective modification only**. It does not alter inference, Prophet fitting, sequence construction, architecture, or the primary Day 1 evaluation protocol.

#### What Does Not Change — Explicit Exclusions

The following are **out of scope** for the Peak-Aware treatment and must not differ from baseline:

| Excluded change | Rationale |
|-----------------|-----------|
| Prophet component | Peak awareness targets GRU residual learning only |
| GRU architecture or hyperparameters (except loss weighting) | Keeps capacity identical for fair comparison |
| Sequence filtering or oversampling | Would change the training set composition |
| Weighted early stopping | Would couple model selection to peak emphasis; baseline uses unweighted `val_loss` |
| Inference-time peak logic | Deployment forecast path must match baseline |
| Preprocessing or data filtering | Near-idle containers remain in both models (documented limitation) |
| Retuning P90 or adding CPU floor | Locked in Phase 2 Task 7 |

#### Scope Boundaries

| In scope (Phase 3–5) | Out of scope |
|----------------------|--------------|
| Timestep-weighted MSE during GRU training | Sequence-weighted MSE |
| Fixed or small-set λ selection (Task 3) | Large hyperparameter grid search |
| Parallel peak-aware modules (Phase 4) | Modifying frozen baseline files |
| Peak / non-peak evaluation reporting (Phase 5) | Changing primary evaluation protocol |
| Comparison vs frozen baseline reference | Re-training or overwriting baseline artifacts |

#### Fairness Verification Plan (Phase 4)

A `scripts/verify_peak_aware_fairness.py` script (planned in Phase 3 Task 5, implemented in Phase 4) will assert:

1. Peak-aware training imports parallel modules only; frozen `utils/hybrid_*.py` files are untouched.
2. Prophet residual generation produces identical `res_mean` and `res_std` given identical input data.
3. Sequence tensors `(X, y)` match baseline for the same `input_window` and data.
4. Model architecture summary matches baseline layer structure.
5. Early stopping monitors unweighted `val_loss` with patience 10.
6. Inference and evaluation call the **unchanged** `hybrid_inference.py` and `hybrid_evaluation.py`.

#### Implementation Decisions

| Decision | Rationale |
|----------|-----------|
| Single experimental variable (training loss) | Standard controlled-experiment design for undergraduate FYP |
| Shared inference/evaluation pipelines | Prevents accidental protocol drift between models |
| Unweighted early stopping | Model selection criterion must match baseline; peak emphasis is a training objective only |
| Peak labels from train-fitted P90 only | Consistent with Phase 2 lock; no validation leakage into threshold definition |
| Document contract before mechanism/λ details | Ensures later design tasks cannot expand scope without explicit revision |

#### Findings

No empirical findings in Task 1. The fairness contract is established as a design artifact.

#### Conclusions

1. The Peak-Aware experiment is a **paired, single-variable comparison** against `experiments/baseline_reference_2026-07-14/`.
2. **Every pipeline stage except GRU training loss** is held identical by contract.
3. Peak awareness applies **only during training**; inference and primary evaluation remain shared with the baseline.
4. The contract is **auditable** via the component table above and the planned Phase 4 fairness verification script.

Task 1 exit criterion is met: the fairness contract is explicit and auditable.

#### Limitations

- Stochastic GRU training may produce small weight differences between runs even with seed 42; both models must use the same seed protocol (specified fully in Task 4).
- The contract assumes Phase 4 implements parallel modules without modifying frozen baseline code; violations would invalidate the comparison.
- Peak-subset metrics (Phase 5) are additional reporting layers; they do not change the core fairness contract for overall MAE/RMSE/MAPE comparison.

#### Next Task

**Task 2 — Confirm Primary Learning Mechanism:** Lock timestep-weighted MSE as the single peak-aware mechanism; document rejected alternatives with Phase 2 evidence; update this document.

---

### Task 2 — Confirm Primary Learning Mechanism

**Status:** Complete  
**Date:** 2026-07-14

#### Objective

Lock **one** peak-aware learning mechanism for the Peak-Aware Hybrid model, consistent with the Task 1 fairness contract. Document rejected alternatives with evidence from Phase 2 so the thesis can defend why timestep-weighted MSE was chosen over other common approaches.

#### Methodology

1. Reviewed Phase 2 findings on timestep frequency (Task 2), sequence structure (Task 3), validation feasibility (Task 5), and the locked peak definition (Task 7).
2. Evaluated candidate peak-aware mechanisms against three design criteria:
   - **Selectivity:** Does the mechanism meaningfully distinguish peak from non-peak training signal?
   - **Fairness:** Does it change only the GRU training objective (Task 1 contract)?
   - **Defensibility:** Is it simple enough to explain and implement for an undergraduate FYP?
3. Selected **timestep-weighted MSE** as the locked mechanism.
4. Recorded rejected alternatives with brief, evidence-based rationale (no implementation).

No code, training, or data analysis was performed in this task.

#### Assumptions

- The GRU predicts a **96-step residual horizon** per training sample (`forecast_horizon = 96`), matching the baseline.
- Peak labels for weighting apply to the **forecast horizon timesteps** in each training window — the same region as the GRU target `y` — not the 96-step input context window.
- Peak timesteps are identified on **real CPU %** using train-fitted per-container P90 thresholds (Phase 2 lock), then mapped to the corresponding residual target steps in each sequence.
- The peak weight λ is a fixed constant (value to be locked in Task 3); mechanism selection is independent of the specific λ choice.

#### Chosen Mechanism (LOCKED)

##### Timestep-weighted MSE

For each training sample *i* with true residual horizon **y**ᵢ ∈ ℝ⁹⁶ and predicted horizon **ŷ**ᵢ ∈ ℝ⁹⁶, assign per-step weights:

```
w_{i,t} = λ   if timestep t in the forecast horizon is a peak (P90 rule on real CPU %)
w_{i,t} = 1   otherwise
```

Training loss:

```
L = (1 / N) Σᵢ (1 / 96) Σₜ w_{i,t} · (y_{i,t} − ŷ_{i,t})²
```

| Property | Specification |
|----------|---------------|
| **Mechanism name** | Timestep-weighted MSE |
| **Weight scope** | 96-step forecast horizon targets only |
| **Peak rule** | `cpu_real >= P90(container_train)` |
| **Non-peak weight** | 1.0 |
| **Peak weight** | λ (fixed; Task 3) |
| **Early stopping** | Unweighted `val_loss` — weights apply to training loss only |
| **Inference** | Unchanged — no weighting at prediction time |

This is the **only** peak-aware mechanism to be implemented in Phase 4.

#### Why Timestep-Weighted MSE (Phase 2 Evidence)

| Evidence source | Finding | Implication for mechanism choice |
|-----------------|---------|-------------------------------|
| **Task 2 — Timestep analysis** | ~17.4% of train timesteps are peaks at P90 | Peaks are frequent enough to benefit from emphasis, but still a minority — timestep weighting can target them selectively |
| **Task 3 — Sequence analysis** | ~96.2% of sequences contain ≥1 peak; only ~3.8% are zero-peak | Sequence-level weighting would up-weight >96% of all sequences — nearly uniform, not selective |
| **Task 3 — Peak steps per sequence** | Mean ~16.3 of 96 horizon steps are peaks at P90 | Peaks are spread within horizons; timestep weighting targets specific steps, not whole sequences |
| **Task 5 — Validation sanity** | ~28.3% val timesteps and ~25.7% Day-1 steps are peaks at P90 | Peak-aware training aligns with an evaluation period that contains abundant peak timesteps |
| **Task 7 — Peak definition lock** | P90 per-container, train-only | Thresholds are fixed; mechanism only needs to apply consistent labels during training |

**Conclusion from evidence:** Timestep weighting provides **finer, meaningful control** over the training objective. Sequence weighting would be a blunt instrument equivalent to near-uniform up-weighting.

#### Comparison to Baseline Loss

| Aspect | Baseline (uniform MSE) | Peak-Aware (timestep-weighted MSE) |
|--------|------------------------|-------------------------------------|
| Per-step weight | Always 1.0 | 1.0 or λ |
| Effective emphasis | Equal on all 96 steps | Higher on ~16–17 of 96 steps (train average) |
| Optimizer behaviour | Minimizes average squared error | Minimizes weighted average squared error |
| Model selection (early stopping) | `val_loss` (unweighted) | **Same** `val_loss` (unweighted) |

When λ = 1, timestep-weighted MSE reduces to baseline uniform MSE (mechanism degenerates to identical loss).

#### Rejected Alternatives

| Alternative | Description | Reason for rejection |
|-------------|-------------|---------------------|
| **Sequence-weighted MSE** | Single weight per sequence: λ if horizon has ≥1 peak, else 1 | **Task 3:** >96% of sequences are peak sequences at P90; would up-weight almost the entire dataset with minimal discrimination between sequences |
| **Oversampling peak sequences** | Duplicate or oversample windows with peak horizons | Changes effective training set composition; violates Task 1 fairness contract (same sequences, same split) |
| **Focal loss** | Down-weights easy examples via `(1 − p_t)^γ` | Introduces a second hyperparameter (γ); not aligned with the single-variable experimental design; harder to defend as a controlled change |
| **Weighted Huber loss** | Robust loss with per-timestep weights | Changes loss **family** (MSE → Huber), not only weighting; confounds peak awareness with robustness to outliers |
| **Quantile / pinball loss** | Predict conditional quantiles | Changes model objective and interpretation; beyond scope of residual-correction hybrid |
| **Prophet-side peak awareness** | Weight or modify Prophet fitting for peaks | Violates scope — peak awareness is a GRU residual training enhancement only |
| **Architecture changes** | Attention, separate peak head, deeper GRU | Changes model capacity; not a training-objective-only modification |
| **Class-balanced sampling** | Batch-level rebalancing of peak vs non-peak steps | Alters stochastic training dynamics and batch composition beyond loss weighting; harder to keep identical to baseline except loss |

#### Implementation Decisions

| Decision | Rationale |
|----------|-----------|
| Lock timestep-weighted MSE in Task 2 (before λ in Task 3) | Mechanism and weight magnitude are separate design choices; mechanism is fully determined by Phase 2 evidence |
| Apply weights to horizon targets only | Matches GRU output shape (96-step residual prediction); input context window is not a training target |
| Keep MSE as the base loss | Minimizes deviation from baseline; only weights change, not loss family |
| Reject all alternatives that change data composition or architecture | Preserves Task 1 single-variable fairness contract |
| Defer λ specification to Task 3 | Mechanism is locked; λ is the remaining hyperparameter for peak emphasis strength |

#### Findings

No new empirical analysis in Task 2. The mechanism choice is fully supported by existing Phase 2 evidence:

1. **Timestep weighting is selective; sequence weighting is not** — the dominant finding from Task 3.
2. **Peak timesteps are a meaningful minority at the timestep level** (~17% train) — sufficient for weighted emphasis without relabelling the majority of data as peaks.
3. **Validation peaks are abundant** — the chosen mechanism targets a behaviour that will be measurable in Phase 5 peak-subset metrics.

#### Conclusions

1. **Timestep-weighted MSE is locked** as the sole peak-aware learning mechanism for this research.
2. The mechanism is consistent with the Task 1 fairness contract: only the GRU training loss changes; early stopping, inference, and evaluation remain identical.
3. Phase 2 evidence provides a clear thesis narrative: sequence-level approaches fail due to >96% peak-sequence prevalence; timestep-level weighting targets the ~17% of timesteps that matter operationally.
4. All considered alternatives are rejected with documented rationale; no further mechanism exploration is planned before Phase 4.

Task 2 exit criterion is met: one mechanism selected with thesis-ready justification.

#### Limitations

- Timestep-weighted MSE assumes peak importance is adequately captured by up-weighting squared residual error; it does not model asymmetric cost (e.g. under-prediction vs over-prediction of peaks differently).
- Near-idle containers with P90 threshold = 0.0 label all horizon steps as peaks (Phase 2 limitation); timestep weighting will up-weight those sequences heavily — documented, not corrected in this study.
- The mechanism does not address Prophet underfitting during peaks; improvement depends on the GRU learning peak-period residuals more accurately.

#### Next Task

**Task 3 — Peak Weight (λ) Decision:** Choose a fixed λ (or select from a small candidate set {3, 5, 10} on sequence-validation only); document justification; update this document.

---

### Task 3 — Peak Weight (λ) Decision

**Status:** Complete  
**Date:** 2026-07-14

#### Objective

Choose and lock a single peak weight **λ** for the timestep-weighted MSE mechanism (Task 2). The value must be justified simply and defensibly for an undergraduate FYP — without a large hyperparameter study or tuning on the temporal validation period.

#### Methodology

1. Defined candidate set **λ ∈ {3, 5, 10}** based on the Phase 3 plan (moderate emphasis range).
2. Computed **analytical weight-balance estimates** using Phase 2 locked statistics:
   - Global train peak timestep fraction: **17.36%** (Task 2)
   - Mean peak steps per 96-step horizon: **16.26** (Task 3, P90)
3. Evaluated each candidate against design criteria:
   - **Sufficient emphasis:** Peak timesteps should meaningfully influence the weighted loss.
   - **Not dominant:** Non-peak timesteps must retain substantial influence to avoid degrading overall accuracy.
   - **Simplicity:** A single fixed λ is preferred over a tuning loop before the primary experiment.
4. Selected **λ = 5** as the locked value.
5. Documented rejected λ-selection approaches (temporal-validation tuning, large grids).

No model training or λ sweep was performed in this task (design phase). The decision is based on Phase 2 data characteristics and loss-balance analysis.

#### Assumptions

- λ applies uniformly to all peak timesteps regardless of container or workload type.
- Non-peak timesteps always have weight **1.0**.
- The primary Peak-Aware experiment (Phase 4–5) uses **one fixed λ**; sensitivity analysis is optional, not required.
- Sequence-validation (80/20 pooled split) is reserved for early stopping only — not for λ selection on the temporal validation period.

#### Candidate Analysis

##### Global timestep balance (train, P90)

Let **p** = fraction of timesteps labelled peak. At P90, **p ≈ 0.174**.

Relative share of **weighted loss** contributed by peak timesteps:

```
peak_loss_share = (p · λ) / (p · λ + (1 − p) · 1)
```

| λ | Peak loss share | Interpretation |
|---|-----------------|----------------|
| **3** | 38.7% | Mild emphasis — peaks influence loss but remain minority |
| **5** | **51.3%** | Balanced — peaks contribute roughly half of weighted loss |
| **10** | 67.8% | Strong emphasis — peaks dominate the objective |

##### Per-sequence balance (mean horizon, P90)

Using mean peak steps **k = 16.26** per 96-step horizon (Task 3):

```
peak_loss_share = (k · λ) / (k · λ + (96 − k))
```

| λ | Peak loss share (per avg sequence) |
|---|-----------------------------------|
| **3** | 37.9% |
| **5** | **50.2%** |
| **10** | 67.1% |

Both views agree: **λ = 5** places peak timesteps at approximately **50% of the weighted training loss** despite being only **~17% of timesteps** — a clear but not extreme emphasis.

#### Intuition — What Does λ Control?

λ controls **how much the model cares about peak timesteps versus normal timesteps** during GRU training:

- Non-peak horizon step weight = **1**
- Peak horizon step weight = **λ**

If a timestep is labelled a peak, its squared residual error counts **λ times more** toward the training loss than a non-peak timestep with the same error magnitude.

The optimizer therefore answers: *"Which mistakes should I fix first?"* — λ determines how strongly peak mistakes are prioritized.

#### Worked Example — One Training Sequence

Consider one baseline 96→96 training window. Under P90 (Phase 2 Task 3), a typical forecast horizon contains approximately **16 peak steps** and **80 non-peak steps**.

Each horizon step contributes `weight × (error)²` to the loss. Using unit error magnitude for illustration, the **relative loss budget** per sequence is:

| λ | Peak contribution (16 steps) | Non-peak contribution (80 steps) | Total | Peak share of loss |
|---|------------------------------|----------------------------------|-------|-------------------|
| **3** | 16 × 3 = **48** | 80 × 1 = **80** | 128 | **37.5%** |
| **5** | 16 × 5 = **80** | 80 × 1 = **80** | 160 | **50.0%** |
| **10** | 16 × 10 = **160** | 80 × 1 = **80** | 240 | **66.7%** |

Peaks are only **~17% of timesteps**, but:

- At **λ = 3**, they drive only **~38%** of the loss — training remains mostly a standard MSE objective.
- At **λ = 5**, they drive **~50%** of the loss — balanced emphasis on peaks and non-peaks.
- At **λ = 10**, they drive **~67%** of the loss — peaks dominate the objective.

#### Worked Example — Equal Error at Peak vs Non-Peak Step

Suppose the model makes the same absolute error at one peak step and one non-peak step: `error = 0.2` (in residual-scaled space).

| λ | Peak step loss term | Non-peak step loss term | Relative importance |
|---|---------------------|-------------------------|---------------------|
| **3** | 3 × 0.2² = **0.12** | 1 × 0.2² = **0.04** | 3:1 |
| **5** | 5 × 0.2² = **0.20** | 1 × 0.2² = **0.04** | 5:1 |
| **10** | 10 × 0.2² = **0.40** | 1 × 0.2² = **0.04** | 10:1 |

At **λ = 5**, fixing a peak error is treated as **five times more important** than fixing the same-sized error at a normal timestep — strong emphasis, but not extreme.

#### Why λ = 5 and Not λ = 3 or λ = 10 (Detailed)

**λ = 3 — too mild for this research contribution**

The stated contribution is *peak-aware* learning. At λ = 3, normal timesteps still contribute **~62%** of the weighted loss. The training objective remains close to uniform MSE, and the peak-aware effect may be too small to demonstrate a clear improvement over the frozen baseline in Phase 5.

*Analogy:* The model is told peaks matter more, but the volume is only turned up slightly — it may barely change behaviour.

**λ = 10 — too aggressive**

At λ = 10, peak timesteps control roughly **two-thirds** of the loss. The model may sacrifice accuracy on the **~83%** of non-peak timesteps. Overall Day-1 MAE/RMSE could worsen even if peak-only metrics improve marginally.

*Analogy:* The model is told to prioritize peaks so strongly that normal CPU periods are neglected — problematic for a forecaster that must perform across the full 24-hour horizon.

**λ = 5 — balanced middle ground**

At λ = 5, peaks and non-peaks each contribute approximately **half** the training loss, despite peaks being only **~17%** of timesteps. This is easy to defend in the thesis:

> *"Peak timesteps are rare (~17% of data) but operationally important, so each peak step receives 5× loss weight, giving peaks ~50% of total training emphasis."*

| λ | Verdict |
|---|---------|
| **3** | Safe, but likely **too weak** to demonstrate a meaningful peak-aware contribution |
| **5** | **Balanced** — peaks matter roughly as much as all non-peaks combined |
| **10** | Likely **too strong** — risks degrading overall forecast quality |

**One-sentence summary:** λ = 5 was chosen because peaks are rare (~17% of timesteps) but operationally important — 5× per-step weight makes them contribute ~50% of the training loss: enough to matter, not enough to break normal forecasting.

#### Decision (LOCKED)

| Parameter | Value |
|-----------|-------|
| **Peak weight λ** | **5** |
| **Non-peak weight** | **1** |
| **Selection method** | Fixed design choice (analytical justification from Phase 2 statistics) |
| **Candidate set considered** | {3, 5, 10} |

**Formal rule:**

> During GRU training, each peak timestep in the 96-step forecast horizon receives loss weight **λ = 5**; all other horizon timesteps receive weight **1.0**.

This value is **locked** for Phase 4 implementation and Phase 5 evaluation.

#### Justification for λ = 5

| Criterion | Rationale |
|-----------|-----------|
| Moderate emphasis | 5× weight is interpretable as "five times more important" without requiring a tuning study |
| Balanced loss contribution | Peaks contribute ~50% of weighted loss at ~17% frequency — strong enough to matter, not so strong that non-peak accuracy is neglected |
| Middle of candidate set | λ = 3 may under-emphasize peaks for a study whose contribution is peak-aware learning; λ = 10 risks over-fitting to peak residuals at the expense of overall MAE |
| Phase 2 alignment | At P90, ~16 of 96 horizon steps are peaks on average — λ = 5 up-weights a meaningful subset without treating the horizon as uniformly peak-dominated |
| Undergraduate scope | Single fixed λ avoids an additional experimental branch; one clear treatment vs baseline |

#### Why Not λ = 3 or λ = 10

| Value | Reason not selected |
|-------|---------------------|
| **λ = 3** | Valid conservative choice, but peak timesteps would still contribute <40% of weighted loss — may produce too small a measurable difference vs baseline for the stated research contribution |
| **λ = 10** | Peak timesteps would contribute ~68% of weighted loss; risk of sacrificing non-peak and overall Day-1 metrics; harder to defend as "moderate" peak awareness |

#### Rejected λ-Selection Approaches

| Approach | Reason for rejection |
|----------|---------------------|
| **Tune λ on temporal validation period** | Would use the Phase 5 evaluation holdout for hyperparameter selection — leakage risk; violates separation of training design from final evaluation |
| **Sequence-val λ sweep in Phase 3 (design only)** | Empirical λ comparison requires training (Phase 4); deferred — analytical selection is sufficient for a single primary experiment |
| **Large grid (e.g. λ ∈ {1, 2, …, 20})** | Unnecessary complexity for undergraduate scope; expands the study beyond one controlled enhancement |
| **Per-container or per-workload λ** | Introduces heterogeneous training objectives; breaks simplicity and fairness narrative |
| **Learned / adaptive λ** | Adds model complexity; not a single fixed experimental variable |

**Note:** An optional λ sensitivity comparison ({3, 5, 10}) may be reported in Phase 6 discussion if Phase 4 training time permits, but the **primary locked experiment uses λ = 5 only**.

#### Implementation Decisions

| Decision | Rationale |
|----------|-----------|
| Fixed λ = 5 (Option A — simplest) | One primary treatment; clear thesis narrative; no λ tuning loop before main experiment |
| Document {3, 5, 10} as considered range | Shows deliberate choice, not arbitrary constant |
| Analytical justification using Phase 2 stats | Design-phase decision without requiring pre-implementation training runs |
| No temporal-validation λ tuning | Protects Phase 5 evaluation integrity |

#### Findings

Analytical analysis only (no training runs):

1. At **λ = 5**, peak timesteps (~17% of train data) contribute **~51%** of the weighted training loss — a balanced emphasis profile.
2. **λ = 3** and **λ = 10** bracket this balance at ~39% and ~68% respectively; λ = 5 is the centre of the pre-specified candidate set.
3. A single fixed λ is consistent with the undergraduate methodology: one mechanism (Task 2) + one weight (Task 3) = one controlled enhancement.

#### Conclusions

1. **λ = 5 is locked** for timestep-weighted MSE in the Peak-Aware Hybrid model.
2. The choice is justified by Phase 2 peak-frequency statistics and loss-balance analysis — no empirical λ sweep required before Phase 4.
3. Temporal-validation tuning and large hyperparameter grids are rejected to preserve evaluation integrity and project scope.
4. Phase 4 will implement `PEAK_WEIGHT = 5` in `utils/peak_config.py` (planned Task 5).

Task 3 exit criterion is met: single λ value locked with written rationale.

#### Limitations

- λ = 5 is not empirically validated on sequence-validation loss before locking; if Phase 4 results show negligible peak-aware effect, λ sensitivity may be noted in Phase 6 only.
- Near-idle containers (all horizon steps labelled peak) receive uniformly high weight under any λ > 1 — a known Phase 2 artefact, not addressed by λ choice.
- The analytical balance assumes average peak fractions; per-container variation is not modelled in this task.

#### Next Task

**Task 4 — Training Protocol Specification:** Define weight tensor shapes, peak labelling alignment with sequences, early stopping, random seed, and threshold artifact storage; update this document.

---

### Task 4 — Training Protocol Specification

**Status:** Complete  
**Date:** 2026-07-14

#### Objective

Specify the **complete Peak-Aware GRU training protocol** with no ambiguity for Phase 4 implementation: where peaks are labelled, how weight tensors align with baseline sequences, how early stopping and random seeds behave, and which artifacts are saved.

#### Methodology

1. Mapped the baseline training pipeline in `utils/hybrid_training.py` and sequence builder in `utils/sequence_utils.py`.
2. Aligned peak-labelling window indices with Phase 2 Task 3 (`scripts/peak_exploration_task3_sequences.py`) — same 96→96 sliding-window logic.
3. Specified weight tensor shapes compatible with Keras `model.fit` and the baseline `y` shape `(n_samples, 96)`.
4. Documented early stopping, seed protocol (from `baseline_metadata.json` and H1.2 ablation script), and threshold artifact schema.
5. Recorded end-to-end training steps in execution order.

No code, training, or data analysis was performed in this task.

#### Assumptions

- Peak-aware training reuses the **same** `train_df.parquet`, Prophet residual generation, global `res_mean`/`res_std`, and `create_residual_sequences()` call as baseline.
- Peak labels for weighting use **real CPU %** (`cpu_real`), not `residual_scaled` — peaks are defined on operational CPU utilization, consistent with Phase 2.
- Per-container P90 thresholds are computed once from the **full train period** per container and stored before sequence weighting.
- The 80/20 pooled sequence split uses the **same `split_idx`** as baseline: `split_idx = int(len(X_all) * 0.8)`.
- Weights apply to **`X_train` / `y_train` only**; `X_val` / `y_val` passed to `validation_data` are unweighted.

#### End-to-End Training Protocol (Execution Order)

| Step | Action | Same as baseline? |
|------|--------|-------------------|
| 1 | Set random seeds (`seed = 42`) | Yes (protocol aligned; baseline notebook did not always set seeds — peak-aware **will**) |
| 2 | Load `data/train_df.parquet` | Yes |
| 3 | `generate_prophet_residuals(global_train)` | Yes |
| 4 | Compute global `res_mean`, `res_std`; append `residual_scaled` | Yes |
| 5 | **Compute and save P90 thresholds** per container (train-only, real CPU %) | **Peak-aware addition** |
| 6 | `create_residual_sequences(...)` → `X_all`, `y_all`, `sample_cids` | Yes |
| 7 | **Build weight matrix `W_all`** aligned with `y_all` (shape `(n_samples, 96)`) | **Peak-aware addition** |
| 8 | Split at `split_idx = int(len(X_all) * 0.8)` → train / sequence-val | Yes |
| 9 | `build_hybrid_gru_model(input_window=96, n_features=1, forecast_horizon=96)` | Yes |
| 10 | `model.compile(optimizer="adam", loss="mse", metrics=["mae"])` | Yes |
| 11 | `model.fit(X_train, y_train, sample_weight=W_train, validation_data=(X_val, y_val), ...)` | **Weighted train loss only** |
| 12 | Early stopping on unweighted `val_loss`, patience 10 | Yes |
| 13 | Save model + `residual_stats.pkl` + peak artifacts to experiment dir | Parallel paths (not `models/`) |

#### 4.1 Where Peaks Are Labelled for Weighting

| Question | Specification |
|----------|---------------|
| **What is weighted?** | Forecast horizon targets **`y`** only — shape `(n_samples, 96)` of `residual_scaled` |
| **What is NOT weighted?** | Input window **`X`** (96-step residual context); sequence-validation loss for early stopping |
| **What defines a peak?** | Real CPU % of the **future timesteps** in each training window |
| **Peak rule** | `cpu_real[t] >= P90(container_train)` |
| **Threshold source** | Train-fitted per-container P90, computed once before sequence generation |
| **Value derivation** | `cpu_real = scaler.inverse_transform(cpu_scaled)` per container |
| **Input window peaks** | **Not used** for weighting — only the 96-step horizon matching `y` |

Peak labels are applied to the **same timestep indices** as the GRU target horizon in `create_residual_sequences()`:

```
For container cid, window start index i:
    X row = residual_scaled[i : i + 96]           # input — not weighted
    y row = residual_scaled[i + 96 : i + 192]     # target — weighted
    weight row = f(cpu_real[i + 96 : i + 192])    # λ on peak steps, 1 otherwise
```

#### 4.2 Weight Matrix Construction

##### Shape and alignment

| Tensor | Shape | Description |
|--------|-------|-------------|
| `X_all` | `(N, 96, 1)` | Baseline input sequences |
| `y_all` | `(N, 96)` | Baseline residual targets |
| `sample_cids` | `(N,)` | Container ID per sequence (from baseline sequence builder) |
| `W_all` | `(N, 96)` | Per-timestep loss weights |
| `W_train` | `(N_train, 96)` | First 80% rows of `W_all` |
| `W_val` | — | **Not passed** to `model.fit` (validation unweighted) |

Where `N` = total pooled sequences (baseline: **41,650** at `input_window=96`).

##### Per-timestep weight rule

For sequence index `s`, horizon step `t ∈ {0, …, 95}`:

```
W_all[s, t] = λ   if cpu_real_future[s, t] >= P90(container = sample_cids[s])
W_all[s, t] = 1   otherwise
```

With **λ = 5** (locked Task 3).

##### Worked indexing example

Container `c_1001`, window start `i = 10` (chronologically sorted train rows):

| Index range | Content | Weighted? |
|-------------|---------|-----------|
| `i … i+95` (rows 10–105) | Input residuals → `X` | No |
| `i+96 … i+191` (rows 106–201) | Target residuals → `y` | **Yes** |
| `cpu_real` at rows 106–201 | Peak labels for `W[s, 0:96]` | λ or 1 per step |

If `cpu_real` at row 110 is 45.2% and `P90(c_1001) = 38.0%`, then `W[s, 4] = 5` (horizon step 4 corresponds to train row `i+96+4`).

##### Implementation note (Phase 4)

Weights will be supplied via Keras `sample_weight=W_train` in `model.fit`, with shape `(batch, 96)` matching `y_train`. If the default MSE reduction does not apply element-wise weights as required, Phase 4 will use an equivalent **custom weighted MSE loss** — same mathematical objective, no change to this protocol.

#### 4.3 P90 Threshold Artifact

Computed **once** before building `W_all`:

```
For each container_id in train_df (chronological sort):
    cpu_real = inverse_minmax(container train cpu_scaled)
    threshold_p90[container_id] = percentile(cpu_real, 90)
```

| Property | Value |
|----------|-------|
| **Fit period** | Train period only |
| **Percentile** | 90 (locked Phase 2) |
| **Storage file** | `peak/peak_thresholds.pkl` (dict: `container_id → float`) |
| **Also saved in** | `peak/peak_config.json` (machine-readable metadata) |
| **Overwrite baseline?** | **No** — saved under `experiments/peak_aware_<timestamp>/` only |
| **Reuse** | Same thresholds for training weights **and** Phase 5 peak-subset evaluation labels |

#### 4.4 Sequence Train / Validation Split

Identical to baseline `train_hybrid_gru()`:

| Parameter | Value |
|-----------|-------|
| Split method | Single chronological split on **pooled** sequences |
| Split index | `split_idx = int(len(X_all) * 0.8)` |
| Training set | `X_all[:split_idx]`, `y_all[:split_idx]`, `W_all[:split_idx]` |
| Sequence-validation set | `X_all[split_idx:]`, `y_all[split_idx:]` (no weights) |
| Purpose of sequence-val | Early stopping only — **not** Phase 5 evaluation |
| Shuffle | `False` |

Phase 5 evaluation uses the **temporal validation period** from preprocessing (`data/val_df.parquet`), not this sequence split.

#### 4.5 `model.fit` Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `optimizer` | `"adam"` | Same as baseline |
| `loss` | `"mse"` | Weighted via `sample_weight` (or equivalent custom loss) |
| `metrics` | `["mae"]` | Same as baseline |
| `epochs` | `100` | Same as baseline |
| `batch_size` | `64` | Same as baseline |
| `shuffle` | `False` | Same as baseline |
| `sample_weight` | `W_train` shape `(N_train, 96)` | **Peak-aware only** |
| `validation_data` | `(X_val, y_val)` — no weights | Unweighted |
| `callbacks` | `[EarlyStopping(...)]` | See §4.6 |
| `verbose` | `0` (script) / configurable | Same convention as ablation script |

#### 4.6 Early Stopping (Unchanged from Baseline)

```python
EarlyStopping(
    monitor="val_loss",       # unweighted MSE on sequence-validation set
    patience=10,
    restore_best_weights=True,
)
```

| Property | Specification |
|----------|---------------|
| Monitor | **`val_loss`** — unweighted |
| Patience | **10** epochs |
| Restore best weights | **True** |
| Peak weights in early stopping | **None** — weights apply to training loss only |

**Rationale:** Model selection must use the same criterion as baseline. Peak emphasis is a training objective modification, not a selection criterion.

#### 4.7 Random Seed Protocol

| Setting | Value |
|---------|-------|
| **Seed** | **42** (per `baseline_metadata.json`) |
| **Python** | `random.seed(42)` |
| **NumPy** | `np.random.seed(42)` |
| **TensorFlow** | `tf.random.set_seed(42)` |
| **When set** | Once at start of training script, before model build and `fit` |
| **Reference implementation** | `scripts/run_hybrid_input_window_ablation.py` → `_set_random_seeds()` |

Both baseline reference training and peak-aware training should document the seed in `training_metadata.json`. Stochastic differences may still occur across hardware; seed protocol maximizes reproducibility.

#### 4.8 Residual Statistics (`res_mean`, `res_std`)

| Property | Specification |
|----------|---------------|
| Computation | Global mean and std of **all train residuals** after Prophet fitting |
| Formula | Same as `train_hybrid_gru()` in `utils/hybrid_training.py` |
| Expected values | `res_mean ≈ 1.54e-05`, `res_std ≈ 0.1112` (baseline reference) |
| Peak-aware requirement | **Must match baseline** given identical `train_df` and Prophet config |
| Storage | `models/residual_stats.pkl` under experiment dir (not overwriting `models/`) |

Peak-aware training does **not** recompute residuals differently; weighting does not alter `res_mean`/`res_std`.

#### 4.9 Phase 4 Artifact Outputs (Training Stage)

Saved under `experiments/peak_aware_<timestamp>/`:

```
experiments/peak_aware_<timestamp>/
├── models/
│   ├── hybrid_gru_peak_aware.keras
│   └── residual_stats.pkl              # {res_mean, res_std} — same computation as baseline
├── peak/
│   ├── peak_thresholds.pkl             # {container_id: threshold_p90}
│   └── peak_config.json                # λ, percentile, rules, locked references
└── training/
    └── training_metadata.json          # seed, hyperparams, split, weighting rule, timestamps
```

Baseline paths (`models/hybrid_gru.keras`, `experiments/baseline_reference_2026-07-14/`) remain **untouched**.

#### Implementation Decisions

| Decision | Rationale |
|----------|-----------|
| Weight horizon targets only | Matches Task 2 mechanism; `y` is what MSE evaluates |
| Label peaks on `cpu_real`, train on `residual_scaled` | Phase 2 lock — operational peak definition, residual learning unchanged |
| `W_all` built with same window indices as `create_residual_sequences` | Guarantees alignment with Phase 2 sequence analysis and baseline tensors |
| Unweighted `validation_data` | Preserves Task 1 fairness contract for early stopping |
| Thresholds saved once as `peak_thresholds.pkl` | Reproducible weights and Phase 5 evaluation labels from one artifact |
| Seed 42 explicitly set in peak-aware script | Baseline metadata records 42; ablation script demonstrates protocol |

#### Findings

No empirical findings in Task 4. The protocol closes all open implementation questions from Tasks 1–3:

1. **Indexing is fully specified** — horizon row `i+96+t` maps to `W[s, t]`.
2. **Tensor shapes are fixed** — `W_train` matches `y_train` as `(N_train, 96)`.
3. **Early stopping and seeds match baseline** — fairness contract preserved.
4. **Artifacts are scoped to experiment directory** — no baseline overwrite.

#### Conclusions

1. The Peak-Aware training protocol is **implementation-ready** for Phase 4.
2. Every step either mirrors `train_hybrid_gru()` exactly or adds a documented peak-aware step (thresholds + weights).
3. No open questions remain on split logic, weight scope, early stopping, or artifact storage.
4. Phase 4 can implement `utils/hybrid_training_peak_aware.py` by following this specification line-for-line.

Task 4 exit criterion is met: training protocol fully specified with no open questions.

#### Limitations

- Element-wise `sample_weight` behaviour depends on Keras MSE reduction; Phase 4 may require a thin custom loss wrapper — mathematically equivalent, not a protocol change.
- Near-idle containers with `threshold_p90 = 0.0` will produce all-λ weight rows for their sequences — known Phase 2 artefact.
- Sequence-validation split is not stratified by peak rate; same as baseline.

#### Next Task

**Task 5 — Module and Artifact Plan:** Document parallel file structure, function responsibilities, and fairness verification script plan; update this document.

---

### Task 5 — Module and Artifact Plan

**Status:** Complete  
**Date:** 2026-07-14

#### Objective

Plan the **parallel code structure** for Phase 4 implementation: new modules, scripts, artifact layout, and JSON schemas — without modifying frozen baseline files. Define the fairness verification script that will assert only the training loss differs.

#### Methodology

1. Reviewed existing baseline modules (`utils/hybrid_*.py`, `utils/sequence_utils.py`) and experiment patterns (`scripts/run_hybrid_input_window_ablation.py`, `scripts/verify_hybrid_*.py`).
2. Mapped Task 4 training protocol steps to module responsibilities.
3. Defined file-level API surface (functions, inputs, outputs) for each new module.
4. Specified experiment directory layout and machine-readable metadata schemas.
5. Designed `verify_peak_aware_fairness.py` checks aligned with Task 1 fairness contract.

No code was written in this task.

#### Assumptions

- Peak-aware code lives in **new files only**; frozen baseline paths are never overwritten.
- New modules may **import read-only** from frozen baseline (e.g. `build_hybrid_gru_model`, `generate_prophet_residuals`, `create_residual_sequences`) — importing is not modifying.
- `hybrid_inference.py` and `hybrid_evaluation.py` are reused **unchanged** for Phase 5 evaluation.
- Experiment outputs follow the timestamped `experiments/` convention used by H1.2 ablation and Phase 2 exploration.

#### Repository Layout After Phase 4

```
FYP_Long_Term/
├── utils/
│   ├── hybrid_config.py                 # FROZEN — baseline
│   ├── hybrid_training.py               # FROZEN — baseline
│   ├── hybrid_inference.py              # FROZEN — shared at eval
│   ├── hybrid_evaluation.py             # FROZEN — shared at eval
│   ├── hybrid_artifacts.py              # FROZEN — baseline production save/load
│   ├── sequence_utils.py                # FROZEN — shared sequence builder
│   ├── peak_config.py                   # NEW — peak-aware constants
│   ├── peak_detection.py                # NEW — thresholds, labels, weights
│   └── hybrid_training_peak_aware.py    # NEW — weighted training pipeline
├── scripts/
│   ├── run_peak_aware_hybrid_experiment.py   # NEW — train + save experiment
│   └── verify_peak_aware_fairness.py         # NEW — fairness assertions
├── models/                              # FROZEN — baseline production (untouched)
└── experiments/
    ├── baseline_reference_2026-07-14/   # FROZEN — control reference
    ├── peak_exploration_2026-07-14/   # FROZEN — Phase 2 outputs
    └── peak_aware_<timestamp>/          # NEW — Phase 4+ outputs
```

#### New Module Specifications

##### `utils/peak_config.py`

Central constants for peak-aware experiments. No training logic.

| Symbol | Value | Source |
|--------|-------|--------|
| `PEAK_PERCENTILE` | `90` | Phase 2 Task 7 lock |
| `PEAK_WEIGHT` | `5` | Phase 3 Task 3 lock |
| `NON_PEAK_WEIGHT` | `1` | Task 2 specification |
| `DEFAULT_INPUT_WINDOW` | `96` | Import or mirror from `hybrid_config` |
| `DAY1_HORIZON` | `96` | Import or mirror from `hybrid_config` |
| `RANDOM_SEED` | `42` | Task 4 / baseline metadata |
| `BASELINE_REFERENCE_DIR` | `experiments/baseline_reference_2026-07-14` | Phase 1 lock |

##### `utils/peak_detection.py`

Peak threshold computation, labelling, and weight-matrix construction.

| Function | Responsibility | Inputs | Outputs |
|----------|----------------|--------|---------|
| `cpu_real_series(group, scaler)` | Inverse MinMax → real CPU % | Container train rows, scaler | `np.ndarray` |
| `compute_peak_thresholds(global_train, scalers, percentile=90)` | Train-only per-container thresholds | Train df, scalers dict | `dict[str, float]` |
| `save_peak_thresholds(thresholds, path)` | Persist threshold artifact | Threshold dict, path | None |
| `load_peak_thresholds(path)` | Load threshold artifact | Path | `dict[str, float]` |
| `label_peak_timesteps(cpu_real, threshold)` | Boolean peak mask | CPU % array, threshold | `np.ndarray` (bool) |
| `build_sequence_weight_matrix(global_train, scalers, thresholds, sample_cids, y_all, lambda_, input_window, forecast_horizon)` | Build `W_all` aligned with `y_all` | Train df, scalers, thresholds, sequence metadata from Task 4 protocol | `np.ndarray` shape `(N, 96)` |

**Design rule:** Weight construction must use the **same window indices** as `create_residual_sequences()` (Task 4 §4.1). Implementation will iterate containers and window starts identically to Phase 2 Task 3.

##### `utils/hybrid_training_peak_aware.py`

Parallel training entry point — mirrors `train_hybrid_gru()` signature where possible.

| Function | Responsibility | Returns |
|----------|----------------|---------|
| `train_hybrid_gru_peak_aware(global_train, scalers, input_window, forecast_horizon, ...)` | Full peak-aware pipeline (Task 4 steps 1–12) | `(model, res_mean, res_std, train_residual_df, peak_thresholds, W_all)` |
| `set_random_seeds(seed=42)` | Python / NumPy / TF seed setup | None |
| `_weighted_mse_loss` (if needed) | Custom loss equivalent to `sample_weight` | Keras loss callable |

**Imports from frozen baseline (read-only):**

- `generate_prophet_residuals` from `utils.hybrid_training`
- `build_hybrid_gru_model` from `utils.hybrid_training`
- `create_residual_sequences` from `utils.sequence_utils`

**Does not import or call:** `train_hybrid_gru()` directly (avoids accidentally using unweighted fit).

##### `scripts/run_peak_aware_hybrid_experiment.py`

End-to-end experiment runner (pattern: `run_hybrid_input_window_ablation.py`).

| Step | Action |
|------|--------|
| 1 | Parse optional CLI args (seed, experiment name override) |
| 2 | `_set_random_seeds(42)` |
| 3 | Create `experiments/peak_aware_<timestamp>/` |
| 4 | Load frozen `data/` artifacts |
| 5 | Call `train_hybrid_gru_peak_aware(...)` |
| 6 | Save all artifacts (see layout below) |
| 7 | Write `experiment_metadata.json` |
| 8 | Print summary paths |

**Does not run Phase 5 evaluation** — evaluation is Phase 5. May optionally print `res_mean`/`res_std` for sanity check.

##### `scripts/verify_peak_aware_fairness.py`

Pre-evaluation fairness gate. Pattern: existing `scripts/verify_hybrid_*.py`.

| Check # | Assertion | Pass criterion |
|---------|-----------|----------------|
| 1 | Frozen file integrity | `utils/hybrid_training.py`, `hybrid_inference.py`, `hybrid_evaluation.py`, `sequence_utils.py` exist and are importable; peak-aware code does not write to `models/` or `baseline_reference/` |
| 2 | Identical residual statistics | `res_mean`, `res_std` from peak-aware training match recomputation via frozen `generate_prophet_residuals()` on same `train_df` (tolerance `1e-6`) |
| 3 | Identical sequences | `X_all`, `y_all` from peak-aware path match baseline `create_residual_sequences()` on same residual df |
| 4 | Identical architecture | `model.summary()` layer names and units match `build_hybrid_gru_model(96, 1, 96)` |
| 5 | Early stopping config | Callback monitors `val_loss`, `patience=10`, `restore_best_weights=True` |
| 6 | Unweighted validation | `model.fit` validation_data has no `sample_weight` |
| 7 | Inference path | Evaluation script loads peak-aware model but calls `hybrid_inference.run_hybrid_inference` and `hybrid_evaluation.evaluate_selected_containers` unchanged |
| 8 | Threshold artifact | `peak_thresholds.pkl` contains one entry per train container; P90 values match `peak_detection.compute_peak_thresholds()` |

**Output:** PASS/FAIL per check; results saved to `experiments/peak_aware_<timestamp>/verification/fairness_check.json`.

#### Frozen Files — Do Not Modify

| Path | Role |
|------|------|
| `utils/hybrid_config.py` | Baseline constants |
| `utils/hybrid_training.py` | Baseline training + shared builders |
| `utils/hybrid_inference.py` | Shared inference (Phase 5) |
| `utils/hybrid_evaluation.py` | Shared evaluation (Phase 5) |
| `utils/hybrid_artifacts.py` | Baseline production save/load |
| `utils/sequence_utils.py` | Shared sequence builder |
| `notebooks/hybrid_model.ipynb` | Baseline notebook |
| `notebooks/preprocessing.ipynb` | Frozen preprocessing |
| `models/hybrid_gru.keras` | Production baseline weights |
| `models/residual_stats.pkl` | Production baseline stats |
| `experiments/baseline_reference_2026-07-14/` | Immutable control reference |

#### Experiment Artifact Layout

```
experiments/peak_aware_<timestamp>/
├── experiment_metadata.json
├── models/
│   ├── hybrid_gru_peak_aware.keras
│   └── residual_stats.pkl
├── peak/
│   ├── peak_thresholds.pkl
│   └── peak_config.json
├── training/
│   └── training_metadata.json
├── verification/
│   └── fairness_check.json              # written by verify script (Phase 4)
└── evaluation/                          # Phase 5 — empty at Phase 4
    ├── evaluation_df.csv
    └── evaluation_summary.csv
```

#### Metadata Schemas

##### `peak/peak_config.json`

```json
{
  "peak_percentile": 90,
  "peak_weight": 5,
  "non_peak_weight": 1,
  "peak_rule": "cpu_real >= P90(container_train)",
  "value_space": "real_cpu_percent",
  "threshold_fit_period": "train_only",
  "weighting_granularity": "timestep",
  "weight_scope": "forecast_horizon_only",
  "phase_2_reference": "experiments/peak_exploration_2026-07-14/peak_definition_decision.json",
  "baseline_reference": "experiments/baseline_reference_2026-07-14"
}
```

##### `training/training_metadata.json`

```json
{
  "model_variant": "hybrid_prophet_gru_peak_aware_v1",
  "input_window": 96,
  "forecast_horizon": 96,
  "optimizer": "adam",
  "loss": "weighted_mse",
  "base_loss": "mse",
  "peak_weight": 5,
  "early_stopping_monitor": "val_loss",
  "early_stopping_patience": 10,
  "epochs_max": 100,
  "batch_size": 64,
  "shuffle": false,
  "random_seed": 42,
  "sequence_val_split": 0.8,
  "res_mean": 0.0,
  "res_std": 0.0,
  "n_sequences_total": 41650,
  "n_sequences_train": 33320,
  "trained_at": "ISO-8601 timestamp",
  "frozen_modules_used": [
    "utils.hybrid_training.generate_prophet_residuals",
    "utils.hybrid_training.build_hybrid_gru_model",
    "utils.sequence_utils.create_residual_sequences"
  ]
}
```

##### `experiment_metadata.json` (experiment root)

```json
{
  "experiment_type": "peak_aware_hybrid",
  "phase": 4,
  "timestamp": "YYYY-MM-DD_HHMMSS",
  "baseline_reference": "experiments/baseline_reference_2026-07-14",
  "peak_definition_reference": "experiments/peak_exploration_2026-07-14/peak_definition_decision.json",
  "design_document": "docs/peak-aware-hybrid/phase-03-learning-design.md",
  "data_artifacts": [
    "data/train_df.parquet",
    "data/val_df.parquet",
    "data/scalers.pkl",
    "data/selected_containers.npy"
  ],
  "outputs": {
    "model": "models/hybrid_gru_peak_aware.keras",
    "residual_stats": "models/residual_stats.pkl",
    "peak_thresholds": "peak/peak_thresholds.pkl",
    "peak_config": "peak/peak_config.json",
    "training_metadata": "training/training_metadata.json"
  },
  "status": {
    "training": "pending",
    "fairness_verification": "pending",
    "evaluation": "pending"
  }
}
```

#### Module Dependency Diagram

```
scripts/run_peak_aware_hybrid_experiment.py
    │
    ├── utils/peak_config.py
    ├── utils/peak_detection.py
    │       └── data/scalers.pkl, data/train_df.parquet
    └── utils/hybrid_training_peak_aware.py
            ├── utils/peak_detection.py
            ├── utils/hybrid_training.py      [READ-ONLY: generate_prophet_residuals, build_hybrid_gru_model]
            └── utils/sequence_utils.py       [READ-ONLY: create_residual_sequences]

scripts/verify_peak_aware_fairness.py
    │
    ├── utils/hybrid_training_peak_aware.py
    ├── utils/hybrid_training.py              [READ-ONLY: baseline comparison]
    ├── utils/sequence_utils.py
    ├── utils/hybrid_inference.py             [READ-ONLY: path check]
    └── utils/hybrid_evaluation.py            [READ-ONLY: path check]

Phase 5 evaluation (future):
    utils/hybrid_evaluation.py
    utils/hybrid_inference.py
    └── loads experiments/peak_aware_<timestamp>/models/*
```

#### Phase 4 Implementation Order

| Order | File | Rationale |
|-------|------|-----------|
| 1 | `utils/peak_config.py` | Constants needed by all other new modules |
| 2 | `utils/peak_detection.py` | Threshold and weight logic; testable in isolation |
| 3 | `utils/hybrid_training_peak_aware.py` | Core training pipeline |
| 4 | `scripts/run_peak_aware_hybrid_experiment.py` | Orchestration and artifact save |
| 5 | `scripts/verify_peak_aware_fairness.py` | Gate before Phase 5 evaluation |

#### Implementation Decisions

| Decision | Rationale |
|----------|-----------|
| Separate `peak_detection.py` from training | Threshold/weight logic is testable and reusable for Phase 5 peak-subset metrics |
| Reuse `build_hybrid_gru_model` from frozen module | Guarantees architecture identity without duplicating layer definitions |
| No `hybrid_artifacts_peak_aware.py` | Experiment runner saves to explicit paths; avoids parallel artifact API surface |
| Fairness script as standalone verifier | Matches baseline verification culture; produces auditable JSON log |
| `hybrid_gru_peak_aware.keras` naming | Distinguishes from baseline `hybrid_gru.keras` in all experiment dirs |
| Phase 5 eval deferred to separate script/phase | Keeps Phase 4 scope to train + verify fairness only |

#### Findings

No empirical findings in Task 5. The module plan closes the gap between Task 4 protocol and Phase 4 coding:

1. **Five new files** — three utils modules and two scripts — sufficient for full implementation.
2. **Zero frozen file modifications** — imports are read-only; outputs go to `experiments/peak_aware_<timestamp>/`.
3. **Eight fairness checks** — directly traceable to Task 1 contract.
4. **JSON schemas** — support reproducibility and thesis appendix without duplicating narrative.

#### Conclusions

1. Phase 4 implementation scope is **fully specified** at the file and function level.
2. Artifact layout and metadata schemas match Task 4 protocol and existing experiment conventions.
3. `verify_peak_aware_fairness.py` provides an auditable gate before Phase 5 comparison.
4. Task 6 can consolidate Tasks 1–5 into the final locked Phase 3 design document.

Task 5 exit criterion is met: file plan and artifact schema documented.

#### Limitations

- Function signatures may gain optional parameters in Phase 4 (e.g. `verbose`) without changing the protocol.
- Fairness check #1 does not hash frozen files by default; a manual git diff remains the authoritative integrity check.
- Phase 5 evaluation script is not specified here — will be planned in Phase 5 or referenced in Task 6 design summary.

#### Next Task

**Task 6 — Lock Phase 3 Design Document:** Consolidate Tasks 1–5 into final design summary, write `phase3_design.json`, mark Phase 3 complete; update this document and overview README.

---

### Task 6 — Lock Phase 3 Design Document

**Status:** Complete  
**Date:** 2026-07-14

#### Objective

Consolidate Tasks 1–5 into a single **locked Phase 3 design** — the handoff document for Phase 4 implementation. Produce machine-readable design metadata and mark Phase 3 complete in the research log.

#### Methodology

1. Reviewed completed Task 1–5 sections in this document.
2. Compiled a final design summary table covering mechanism, λ, protocol, modules, and fairness contract.
3. Wrote `experiments/peak_aware_design_2026-07-14/phase3_design.json` as the machine-readable lock file.
4. Wrote `experiments/peak_aware_design_2026-07-14/phase3_design.md` as a short decision memo.
5. Updated overview README to mark Phase 3 complete.

No implementation, training, or baseline modification was performed.

#### Locked Phase 3 Design Summary

| Design element | Locked choice |
|----------------|---------------|
| **Experimental variable** | GRU training loss only |
| **Mechanism** | Timestep-weighted MSE (Task 2) |
| **Peak definition** | P90 per-container, train-only, real CPU % (Phase 2) |
| **Peak weight λ** | **5** (Task 3) |
| **Non-peak weight** | **1** |
| **Weight scope** | 96-step forecast horizon targets only (Task 4) |
| **Early stopping** | Unweighted `val_loss`, patience 10 (Task 4) |
| **Random seed** | 42 (Task 4) |
| **Input window** | 96 |
| **Forecast horizon** | 96 |
| **Implementation** | Parallel modules only (Task 5) |
| **Inference / evaluation** | Unchanged `hybrid_inference.py`, `hybrid_evaluation.py` |
| **Baseline control** | `experiments/baseline_reference_2026-07-14/` |

#### Fairness Checklist (Locked)

| # | Requirement | Source task |
|---|-------------|-------------|
| 1 | Only GRU training loss differs vs baseline | Task 1 |
| 2 | Preprocessing, Prophet, sequences, architecture identical | Task 1 |
| 3 | Timestep-weighted MSE — sole mechanism | Task 2 |
| 4 | λ = 5 fixed; no temporal-validation λ tuning | Task 3 |
| 5 | Weights on `y_train` only; unweighted early stopping | Task 4 |
| 6 | P90 thresholds saved once; reused for Phase 5 labels | Task 4 |
| 7 | Five new files; zero frozen file modifications | Task 5 |
| 8 | Fairness verification script before Phase 5 eval | Task 5 |

#### Phase 5 Evaluation Preview (From Locked Design)

Phase 5 will compare Peak-Aware vs frozen baseline using:

| Metric tier | Metrics | Label rule |
|-------------|---------|------------|
| **Overall** | Day-1 MAE, RMSE, MAPE (99 containers) | All validation timesteps |
| **Peak subset** | Peak-only MAE, RMSE | P90 rule on validation CPU % (train-fitted thresholds) |
| **Non-peak subset** | Non-peak MAE, RMSE | Complement of peak subset |
| **Optional stratification** | By workload type (`stable` / `medium` / `spiky`) | Same P90 rule; interpret spiky cautiously (Phase 2) |

Evaluation protocol remains identical to baseline; Peak-Aware model is loaded from `experiments/peak_aware_<timestamp>/models/` and evaluated through shared inference/evaluation modules.

#### Task Completion Record

| Task | Outcome |
|------|---------|
| 1. Fairness contract | Paired single-variable comparison documented |
| 2. Learning mechanism | Timestep-weighted MSE locked; 8 alternatives rejected |
| 3. Peak weight λ | λ = 5 locked with analytical justification |
| 4. Training protocol | 13-step protocol; weight shapes and artifacts specified |
| 5. Module plan | 5 new files; 8 fairness checks; JSON schemas |
| 6. Design lock | `phase3_design.json` written; Phase 3 complete |

#### Generated Outputs

| Output | Location |
|--------|----------|
| Machine-readable design lock | `experiments/peak_aware_design_2026-07-14/phase3_design.json` |
| Design decision memo | `experiments/peak_aware_design_2026-07-14/phase3_design.md` |
| Full research log (Tasks 1–6) | `docs/peak-aware-hybrid/phase-03-learning-design.md` |

#### Known Limitations (Carry Forward to Phase 4–6)

1. **Near-idle zero-threshold artefact** — five containers contribute ~29% of train peak labels at P90 (Phase 2).
2. **Train/validation peak rate divergence** — validation peaks ~28% vs train ~17% under fixed thresholds.
3. **Spiky workload mislabelling** — high-CV near-idle containers inflate spiky-group statistics.
4. **λ not empirically swept** — λ = 5 chosen analytically; optional sensitivity noted for Phase 6 only.
5. **Stochastic training** — seed 42 maximizes reproducibility; hardware may still introduce minor variance.

#### Conclusions

Phase 3 is **complete**. The Peak-Aware Hybrid design is fully specified and locked:

1. **One mechanism** — timestep-weighted MSE with λ = 5.
2. **One experimental variable** — GRU training loss vs frozen baseline.
3. **Implementation-ready** — module plan, protocol, artifacts, and verification script specified for Phase 4.
4. **Evaluation-ready** — Phase 5 metric tiers previewed using the same P90 definition.

**Next phase:** Phase 4 — Implementation (awaiting approval to begin).

---

## Phase 3 Summary

Phase 3 Peak-Aware Learning Design ran from 2026-07-14 across six tasks (design only — no code):

| Task | Outcome |
|------|---------|
| 1. Fairness contract | Single-variable paired comparison vs `baseline_reference_2026-07-14` |
| 2. Learning mechanism | Timestep-weighted MSE locked |
| 3. Peak weight λ | λ = 5 locked (candidates {3, 5, 10} considered) |
| 4. Training protocol | Weight alignment, early stopping, seeds, artifacts specified |
| 5. Module plan | 5 new files + 8 fairness checks + experiment layout |
| 6. Design lock | `phase3_design.json` — ready for Phase 4 |

**Locked design:** Timestep-weighted MSE, P90 peaks, λ = 5, unweighted early stopping, parallel modules, shared inference/evaluation.

**Next phase:** Phase 4 — Implementation (awaiting approval).

