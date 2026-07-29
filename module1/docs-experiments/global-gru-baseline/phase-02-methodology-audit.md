# Phase 2 — Methodology Audit & Fair Comparison

[← Overview](README.md) · [Phase 1 Planning](phase-01-research-planning.md)

---

## 3. Phase 2 — Methodology Audit & Fair Comparison

### 3.1 Objective

Document the **current state** of the Global GRU prototype, identify methodological defects that invalidate cross-methodology comparison, and establish a **formal fairness contract** between the frozen Hybrid control and the proposed Global GRU baseline.

This phase is **audit and design documentation only**: no implementation, no training, no baseline modification.

### 3.2 Methodology

1. Compared `notebooks/global_gru_model.ipynb` against Module 1 research design (`.cursor/skills/dracasys/SKILL.md`).
2. Cross-referenced findings with `docs/methodology-gap-analysis.md` and `docs/recommended-evaluation-protocol.md`.
3. Mapped Hybrid baseline evaluation protocol from `experiments/baseline_reference_2026-07-14/config/baseline_metadata.json`.
4. Documented component-by-component fairness contract for Global GRU v1.
5. Recorded risks and pre-coding verification requirements.

No code, training, or data analysis was performed in this phase.

### 3.3 Assumptions

- The methodology gap analysis (July 2026) accurately describes the Global GRU notebook defects.
- The Hybrid baseline evaluation protocol is the authoritative standard for fair comparison.
- Fixing validation-target leakage is a **methodological correction**, not an accuracy optimisation that confounds the baseline experiment.
- Internal model differences between the two methodologies (layer sizes, features, training hyperparameters) are **consequences of each forecasting approach** and must be documented, not treated as the primary experimental variable.

---

## 3.4 Current Global GRU Prototype — Issue Register

**Source:** `notebooks/global_gru_model.ipynb`, `docs/methodology-gap-analysis.md`

### Critical — Validation-Target Leakage

```python
# Current (INCORRECT)
full_df = pd.concat([global_train, global_val])
X_all, y_all, cids_all = create_longterm_sequences(full_df, ...)
split_idx = int(len(X_all) * 0.8)
X_train = X_all[:split_idx]   # ~21% of sequences target val period
```

| Sequence type | Count | Percentage |
|---------------|-------|------------|
| Target entirely in train period | 42,172 | 73.4% |
| Target crosses train/val boundary | 9,500 | 16.5% |
| Target entirely in val period | 5,773 | 10.0% |

**26.6% of sequences have targets in the validation period.** With an 80/20 index split, approximately **21% of all sequences** can enter GRU training — this is validation-target leakage.

### High — Incompatible Evaluation Protocol

| Aspect | Hybrid (frozen) | Global GRU (current notebook) |
|--------|-----------------|-------------------------------|
| Final evaluation | Temporal holdout on `global_val` | Sequence index split on train+val |
| Demo container | `c_11461` | `c_11307` |
| Metrics space | Real CPU % | Mixed (scaled MAE ~0.016 reported) |
| Multi-container aggregate | 99 containers | Partial / single-container demo |
| MAPE | Computed (ε = 0.01) | Not implemented |

### Medium — Engineering Gaps

| Issue | Detail |
|-------|--------|
| Model not persisted | No `models/global_gru.keras` |
| Duplicate sequence function | `create_longterm_sequences` inline, not in `sequence_utils.py` |
| No verification scripts | Unlike Hybrid's five-script suite |
| No random seeds | Non-reproducible training |
| Mislabeled variables | `global_model`, `hybrid_mae` in Global GRU notebook |
| No experiment logging | No timestamped output directories |

### Low — Documented Methodology Differences (By Design)

These are **consequences of the Global GRU forecasting methodology**, not defects:

| Item | Hybrid methodology | Global GRU methodology |
|------|-------------------|------------------------|
| Input window | 96 (locked after H1.2 ablation) | 96 |
| Model input features | 1 (`residual_scaled`) | 3 (`cpu_scaled`, `cpu_mean`, `cpu_std`) |
| Model layer configuration | 256→128→64 + Dropout(0.2) | 128→64 + Dropout(0.2) after first GRU |
| Training epochs | 100 max, patience 10 | 50 max, patience 3 |
| Batch size | 64 | 256 |
| Forecasting task | Residual correction | Direct CPU forecasting |

**Global GRU notebook architecture (source of truth):**

```
GRU(128, return_sequences=True) → Dropout(0.2) → GRU(64) → Dense(64, relu) → Dense(96)
```

These differences must be documented in thesis comparison as properties of each **forecasting methodology**, not as a controlled architecture ablation.

---

## 3.5 Required Methodology Fixes (Phase 4 Scope)

| Fix | Current | Required |
|-----|---------|----------|
| Sequence source | `concat(train, val)` | `global_train` only |
| Training val split | 80/20 on pooled train+val sequences | 80/20 on train sequences only (early stopping) |
| Final evaluation | Sequence index split | Temporal holdout: last 96 train → first 96 val |
| Metrics | Scaled or inconsistent | Real CPU % (MAE, RMSE, MAPE) |
| Cohort | Different demo container | Same 99 evaluable containers as Hybrid |
| Artifacts | None saved | Model + metadata + timestamped experiment dir |

---

## 3.6 Fair Comparison Contract

### The Controlled Experiment

| Property | Specification |
|----------|---------------|
| **Control** | Frozen Hybrid Prophet + GRU (`hybrid_prophet_gru_v1`) |
| **Baseline** | Global GRU v1 (standard baseline, no Peak-Aware) |
| **Intentional difference** | **Forecasting methodology only** |
| **Comparison type** | Forecasting methodology comparison under identical data, cohort, period, and metrics |
| **Baseline metrics (Day 1)** | MAE 1.746, RMSE 2.388, MAPE 111.94 (99 containers) |

### What Is Identical vs Different

| Component | Hybrid (control) | Global GRU (baseline) | Status |
|-----------|------------------|----------------------|--------|
| **Preprocessing pipeline** | Frozen `data/` artifacts | Same frozen artifacts | **Identical** |
| **Temporal data split** | Chronological holdout from preprocessing | Same split | **Identical** |
| **Container cohort** | 100 selected; 99 evaluable | Same cohort | **Identical** |
| **CPU scaling** | Per-container MinMax on `cpu_scaled` | Same scalers | **Identical** |
| **Evaluation procedure** | Day 1 temporal holdout | Same protocol | **Identical** |
| **Primary horizon** | Day 1 — 96 steps (24 h) | Same | **Identical** |
| **Metrics & reporting** | MAE, RMSE, MAPE in real CPU % | Same formulas, same MAPE ε | **Identical** |
| **Aggregation** | Per-container → mean ± std | Same | **Identical** |
| **Random seed** | 42 | 42 | **Identical** |
| **Forecasting methodology** | Prophet + GRU residuals | Direct GRU on CPU | **Different (by design)** |
| **Peak-Aware Learning** | Not in baseline control | Not in Global GRU baseline | **Identical (absent)** |

Internal model configuration (layer sizes, feature count, training hyperparameters) differs as a **consequence** of each methodology — it is not the primary variable under test.

### Fair Comparison Principles

| Principle | Requirement |
|-----------|-------------|
| Compare methodologies, not architectures | Research question is which forecasting approach performs better under fair conditions |
| Frozen preprocessing | No re-running `preprocessing.ipynb` |
| No validation-target leakage | Global GRU trains on train-period targets only |
| Train/eval separation | Evaluation never calls `model.fit()` |
| Real CPU % metrics | Never compare scaled MAE to Hybrid real-% MAE |
| Non-overwriting artifacts | Timestamped experiment directories |
| Reuse-first code | Shared utilities reused; only Global GRU–specific logic in new modules |
| Single methodology variable | Only the forecasting methodology differs at comparison time |

---

## 3.7 Three-Layer Split Model (Target State)

```
Layer 1 — Preprocessing (frozen):
  Per-container 80/20 chronological split → train_df / val_df

Layer 2 — GRU training (Global GRU):
  Sequences from train period only
  80/20 index split on pooled train sequences → early stopping only

Layer 3 — Final evaluation (Global GRU):
  Per-container temporal holdout on val_df
  Last 96 train steps → predict first 96 val steps
```

This mirrors the Hybrid baseline's train/eval separation documented in `docs/hybrid-prophet-gru.md` §3.6 and §4.

---

## 3.8 Risks Register

| Risk | Severity | Mitigation |
|------|----------|------------|
| Validation-target leakage | **Critical** | `verify_global_gru_data_split.py` before training |
| Evaluating on wrong period | **High** | Canonical single Day 1 forecast per container |
| Scaled-space metrics | **High** | Always inverse-transform before metrics |
| Modifying frozen Hybrid code | **High** | Reuse read-only imports; new modules for Global GRU only |
| Overwriting Hybrid artifacts | **High** | Separate `models/global_gru.keras` |
| Unnecessary code duplication | **Medium** | Reuse shared metrics and sequence utilities |
| Single-window vs sliding-window eval | **Medium** | One Day 1 forecast per container (match Hybrid) |
| Stochastic training | **Medium** | Seed 42; document hardware dependency |
| MAPE instability | **Medium** | `MAPE_EPSILON = 0.01` (same as Hybrid) |

---

## 3.9 Pre-Coding Verification Checklist

Before Phase 4 implementation begins (after Phase 0.5 approval), confirm:

- [ ] [Phase 0.5 Baseline Specification](phase-03-5-baseline-specification.md) reviewed and approved
- [ ] `data/scalers.pkl` exists and loads for all selected containers
- [ ] `selected_containers.npy` contains 100 IDs (including stale `c_14674`)
- [ ] Train/val row counts per container: ≥ 96 train and ≥ 96 val steps
- [ ] Hybrid baseline reproducible via `scripts/verify_hybrid_evaluation.py`
- [ ] No Peak-Aware imports in Global GRU modules
- [ ] Sequence count from `global_train` only ≈ 42K (not ~58K from concat)

---

## 3.10 Task Plan

| Task | Focus | Status |
|------|-------|--------|
| 1 | Document current prototype defects | **Complete** |
| 2 | Map leakage mechanism with sequence statistics | **Complete** |
| 3 | Write fairness contract vs Hybrid control | **Complete** |
| 4 | Define three-layer split model | **Complete** |
| 5 | Record risks and pre-coding checklist | **Complete** |

---

## 3.11 Exit Criteria (Phase 2 Complete)

- [x] Issue register documented with severity
- [x] Required methodology fixes enumerated
- [x] Fairness contract written (methodology comparison, not architecture comparison)
- [x] Three-layer split model defined
- [x] Risks and pre-coding checklist recorded
- [x] Methodology differences documented as design consequences, not hidden confounds

### 3.12 Conclusion

Phase 2 is complete. The Global GRU prototype has critical validation-target leakage and an incompatible evaluation protocol. A formal fairness contract defines what must be identical vs intentionally different when comparing **forecasting methodologies** against the frozen Hybrid control.

**Next phase:** [Phase 3 — Pipeline & Architecture Design](phase-03-pipeline-design.md)
