# Phase 1 — Research Planning & Scope

[← Overview](README.md)

---

## 2. Phase 1 — Research Planning & Scope

### 2.1 Objective

Define the research scope, success criteria, comparison philosophy, and phased workflow for establishing a **standard Global GRU baseline** before any Peak-Aware Learning or cross-methodology thesis results are reported.

This phase is **planning only**: no implementation, no training, no modification of frozen Hybrid or preprocessing artifacts.

### 2.2 Methodology (Phase 1)

Phase 1 followed a structured planning workflow aligned with the Peak-Aware Hybrid research log (`docs/peak-aware-hybrid/`):

1. Reviewed all relevant project documentation, Cursor skills, and the frozen Hybrid control reference.
2. Audited the existing Global GRU notebook prototype against Module 1 research design.
3. Documented the comparison philosophy — forecasting methodologies under identical protocol, not neural architecture comparison.
4. Defined research objectives, questions, success criteria, and explicit non-goals.
5. Designed the full phased workflow including the Phase 0.5 specification gate before implementation.
6. Created the `docs/global-gru-baseline/` research log with README index and phase documents.

No code, training, or data analysis was performed in this phase.

### 2.3 Assumptions

- The Hybrid Prophet + GRU baseline in `experiments/baseline_reference_2026-07-14/` is finalized and must not be modified.
- Frozen preprocessing artifacts in `data/` are shared by all forecasting methodologies and must not be changed for Global GRU baseline establishment.
- Peak-Aware Hybrid research (complete through Phase 5) is a **separate study track** on the Hybrid methodology only; it does not substitute for a Global GRU baseline.
- The existing Global GRU notebook model form (128→64 GRU, 3 input features) represents the intended starting configuration and should not be changed during methodology fixes.
- Research Objective 2 (*compare Hybrid Prophet + GRU against Global GRU*) requires identical evaluation protocol before any thesis comparison table is valid.

### 2.4 Research Objective

**Primary:** Implement and freeze **Global GRU v1** as a defensible baseline that forecasts `cpu_util_percent` directly using a single shared GRU trained across many containers.

**Research questions:**

| Question | How the baseline addresses it |
|----------|-------------------------------|
| Can a single global model generalize across diverse container workloads? | One GRU trained on pooled sequences from 100 containers |
| How does the Global GRU forecasting methodology compare to Hybrid Prophet + GRU? | Fair Day 1 evaluation on identical cohort, period, and metrics |
| Is Global GRU suitable for later Peak-Aware extensions? | Frozen baseline with reproducible artifacts before any enhancement |

### 2.5 Comparison Philosophy

This study compares **forecasting methodologies** — not neural network architectures in isolation.

| Fixed (identical across methodologies) | Differs by design |
|----------------------------------------|-------------------|
| Preprocessing pipeline | Forecasting methodology |
| Temporal data split | |
| Evaluation procedure | |
| Metrics and reporting | |
| Container cohort | |

The Hybrid methodology decomposes forecasting (Prophet + GRU residuals). The Global GRU methodology forecasts CPU directly with a single shared model. Internal model configuration differences are consequences of each methodology, not the primary experimental variable.

### 2.6 Success Criteria (Study-Level)

The Global GRU baseline study is complete when:

- [ ] No validation-target leakage in training sequences
- [ ] Day 1 metrics reported on 99 evaluable containers (same cohort as Hybrid)
- [ ] MAE, RMSE, MAPE computed in **real CPU %** after inverse MinMax
- [ ] Model and metadata saved to timestamped experiment directory
- [ ] All verification scripts pass
- [ ] Frozen reference copied to `experiments/global_gru_reference_*/`
- [ ] Side-by-side Hybrid vs Global GRU methodology comparison table is defensible for thesis

### 2.7 Explicit Non-Goals

| Item | Rationale |
|------|-----------|
| Peak-Aware Learning | Deferred until Global GRU baseline is frozen |
| Preprocessing changes | Would confound methodology comparison |
| Hyperparameter tuning / model optimisation | Baseline uses locked notebook values |
| Multi-component changes | Single-variable research discipline |
| Unseen container holdout | Follow-on experiment after baseline freeze |
| Feature or layer ablations | Separate experiments; not part of v1 baseline |

### 2.8 Phased Workflow (Locked in Task 6)

| Phase | Name | Purpose |
|-------|------|---------|
| 1 | Research Planning & Scope | **This document** — scope, criteria, workflow |
| 2 | Methodology Audit & Fair Comparison | Document gaps, fairness contract vs Hybrid |
| 3 | Pipeline & Architecture Design | Design decisions before specification lock |
| **0.5** | **Global GRU Baseline Specification** | Formal research contract — gate before coding |
| 4 | Implementation | Minimal Global GRU modules, notebook refactor |
| 5 | Evaluation & Verification | Multi-container Day 1 eval, verification scripts |
| 6 | Baseline Freeze & Methodology Comparison | Immutable reference + thesis comparison |

### 2.9 Relationship to Other Research Tracks

```
Module 1 Forecasting Methodologies
├── Hybrid Prophet + GRU
│   ├── Baseline (FROZEN) ───────────── baseline_reference_2026-07-14/
│   └── Peak-Aware Hybrid (COMPLETE) ── peak_aware_2026-07-14_164518/
│
└── Global GRU
    ├── Standard Baseline (THIS STUDY) ── global_gru_reference_* (planned)
    └── Peak-Aware Global GRU ────────── deferred
```

### 2.10 Task Plan

| Task | Focus | Status |
|------|-------|--------|
| 1 | Review project documentation, skills, and Hybrid control | **Complete** |
| 2 | Audit Global GRU notebook prototype | **Complete** |
| 3 | Document comparison philosophy and fair experimentation principles | **Complete** |
| 4 | Define research objectives, questions, and success criteria | **Complete** |
| 5 | Define explicit non-goals and implementation philosophy | **Complete** |
| 6 | Design phased research workflow | **Complete** |
| 7 | Create research documentation folder and phase index | **Complete** |
| 8 | Lock Phase 1 research planning record | **Complete** |

### 2.11 Exit Criteria (Phase 1 Complete)

- [x] Research objective and questions documented
- [x] Comparison philosophy documented (methodologies, not architectures)
- [x] Success criteria defined
- [x] Non-goals and implementation philosophy explicit
- [x] Phased workflow locked (including Phase 0.5 gate)
- [x] Relationship to Hybrid and Peak-Aware tracks documented
- [x] `docs/global-gru-baseline/` folder created with README index
- [x] All eight Phase 1 tasks documented in this file

---

## Phase 1 Task Details

---

### Task 1 — Review Project Documentation, Skills, and Hybrid Control

**Status:** Complete  
**Date:** 2026-07-17

#### Objective

Establish a complete understanding of the project research context, coding standards, experiment conventions, and the frozen Hybrid control baseline before planning the Global GRU baseline study.

#### Methodology

1. Reviewed Cursor skills: `dracasys/`, `experiments/`, `python-coding-standards/`, `visualization/`.
2. Reviewed project documentation index (`docs/README.md`) and supporting references:
   - `docs/hybrid-prophet-gru.md`
   - `docs/data-pipeline-reference.md`
   - `docs/methodology-gap-analysis.md`
   - `docs/recommended-evaluation-protocol.md`
   - `docs/implementation-roadmap.md`
3. Reviewed Peak-Aware Hybrid research log structure (`docs/peak-aware-hybrid/`) as the documentation template for this study.
4. Read frozen Hybrid control metadata: `experiments/baseline_reference_2026-07-14/config/baseline_metadata.json`.
5. Recorded Hybrid Day 1 control metrics (99 containers, real CPU %).

#### Implementation Decisions

| Decision | Rationale |
|----------|-----------|
| Use Peak-Aware Hybrid doc structure as template | Consistent research log format across Module 1 studies |
| Treat `baseline_reference_2026-07-14/` as permanent Hybrid control | Matches Peak-Aware study fairness contract |
| Adopt experiments skill conventions (seeds, artifacts, no overwrite) | Ensures Global GRU baseline is reproducible |
| Module 1 scope locked to `cpu_util_percent` only | Consistent with DracaSys skill and frozen preprocessing |

#### Generated Outputs

| Output | Location |
|--------|----------|
| Hybrid control metrics reference | §2.5 in [README](README.md) |
| Documentation source inventory | Task 1 methodology above |
| Planning assumptions list | §2.3 of this document |

#### Findings

| Finding | Detail |
|---------|--------|
| Hybrid baseline is verified and frozen | Five verification scripts passed 2026-07-14 |
| Global GRU comparison is not yet valid | Documented in `methodology-gap-analysis.md` |
| Peak-Aware Hybrid is Hybrid-only | Does not replace need for Global GRU baseline |
| Recommended evaluation protocol exists | Target state defined; Global GRU must align |

#### Conclusions

Task 1 is complete. The project research context, standards, and Hybrid control reference are understood and recorded as inputs to all subsequent Global GRU baseline phases.

#### Limitations

- No hands-on verification of Hybrid scripts was run during Task 1 (planning only).
- Global GRU notebook was not yet audited in detail — deferred to Task 2.

#### Next Task

**Task 2 — Audit Global GRU Notebook Prototype:** Read `notebooks/global_gru_model.ipynb`; identify methodological defects; record engineering gaps blocking fair methodology comparison.

---

### Task 2 — Audit Global GRU Notebook Prototype

**Status:** Complete  
**Date:** 2026-07-17

#### Objective

Audit the existing `notebooks/global_gru_model.ipynb` prototype against Module 1 research design and document all defects that prevent a fair forecasting methodology comparison with the frozen Hybrid control.

#### Methodology

1. Read `notebooks/global_gru_model.ipynb` end-to-end.
2. Cross-referenced behaviour with `docs/methodology-gap-analysis.md` and `docs/data-pipeline-reference.md`.
3. Identified sequence generation source, train/val split method, and evaluation protocol.
4. Compared prototype behaviour against Hybrid baseline protocol in `docs/hybrid-prophet-gru.md`.
5. Classified issues by severity: Critical / High / Medium / Low.

#### Assumptions

- The methodology gap analysis (July 2026) accurately describes known Global GRU defects.
- The notebook represents the starting point for Global GRU v1 model form (128→64, 3 features) — not an optimised configuration.

#### Implementation Decisions

| Decision | Rationale |
|----------|-----------|
| Treat leakage fix as methodological correction | Not an accuracy optimisation that confounds baseline |
| Preserve existing notebook model form for v1 | Avoid changing multiple variables at once |
| Document methodology differences separately from defects | Layer sizes differ by design; leakage is a defect |

#### Findings

**Critical — validation-target leakage:**

```python
full_df = pd.concat([global_train, global_val])  # INCORRECT
# ~21% of training sequences can target the validation period
```

**High — incompatible evaluation protocol:**

| Aspect | Hybrid (frozen) | Global GRU (notebook) |
|--------|-----------------|----------------------|
| Final evaluation | Temporal holdout | Sequence index split on train+val |
| Demo container | `c_11461` | `c_11307` |
| Metrics | Real CPU % | Mixed / scaled MAE reported |
| Multi-container | 99 containers | Single-container demo |

**Medium — engineering gaps:** model not persisted, duplicate sequence function, no verification scripts, no random seeds, mislabeled variables.

**Low — methodology differences (by design):** different GRU form, features, and hyperparameters as consequences of the Global GRU forecasting approach.

#### Generated Outputs

| Output | Location |
|--------|----------|
| Issue register (detailed) | [Phase 2 — Methodology Audit](phase-02-methodology-audit.md) §3.4 |
| Required fixes list | [Phase 2](phase-02-methodology-audit.md) §3.5 |

#### Conclusions

Task 2 is complete. The Global GRU prototype cannot support thesis-quality methodology comparison until leakage is fixed and evaluation is aligned with the Hybrid protocol. Detailed audit output is expanded in Phase 2.

#### Limitations

- Audit was documentation-based; no training run was executed to confirm sequence counts empirically during Task 2 (verified in Phase 2 register from prior analysis).

#### Next Task

**Task 3 — Document Comparison Philosophy:** Formalise that this study compares forecasting methodologies under identical protocol; define what is fixed vs what differs by design.

---

### Task 3 — Document Comparison Philosophy and Fair Experimentation Principles

**Status:** Complete  
**Date:** 2026-07-17

#### Objective

Define the formal comparison philosophy for the Global GRU baseline study: a **forecasting methodology comparison** under an identical experimental protocol — not a neural network architecture comparison.

#### Methodology

1. Reviewed Module 1 research objectives in `.cursor/skills/dracasys/SKILL.md`.
2. Mapped fixed experimental components vs intentional methodology differences.
3. Drafted fair comparison principles aligned with Peak-Aware Hybrid fairness contract (`docs/peak-aware-hybrid/phase-03-learning-design.md` Task 1).
4. Documented reuse-first code philosophy (avoid unnecessary Hybrid module duplication).

#### Implementation Decisions

| Decision | Rationale |
|----------|-----------|
| Compare methodologies, not GRU layer configs | Primary research question is Hybrid vs Global GRU approach |
| Fix preprocessing, split, eval, metrics across both | Enables defensible thesis comparison |
| Reuse shared utilities; new code for Global GRU–specific logic only | Maintainable research codebase |
| Single methodology variable at comparison time | Scientific validity |

#### Fair Comparison Contract (Summary)

| Component | Status |
|-----------|--------|
| Preprocessing pipeline | **Identical** |
| Temporal data split | **Identical** |
| Evaluation procedure | **Identical** |
| Metrics and reporting | **Identical** |
| Container cohort | **Identical** |
| Forecasting methodology | **Different by design** |
| Peak-Aware Learning | **Absent in both baselines** |

#### Generated Outputs

| Output | Location |
|--------|----------|
| Comparison philosophy (overview) | [README](README.md) §1.4 |
| Full fairness contract | [Phase 2](phase-02-methodology-audit.md) §3.6 |
| Code reuse philosophy | [README](README.md) §1.6 |

#### Conclusions

Task 3 is complete. The study is framed as a forecasting methodology comparison with a documented fairness contract. This framing applies to all subsequent phases and thesis reporting.

#### Next Task

**Task 4 — Define Research Objectives, Questions, and Success Criteria.**

---

### Task 4 — Define Research Objectives, Questions, and Success Criteria

**Status:** Complete  
**Date:** 2026-07-17

#### Objective

Document the primary research objective, supporting research questions, and measurable success criteria for the Global GRU baseline study.

#### Methodology

1. Mapped objectives to Module 1 Research Objective 2 (Hybrid vs Global GRU comparison).
2. Defined three supporting research questions tied to generalization, methodology comparison, and Peak-Aware readiness.
3. Specified measurable success criteria aligned with Hybrid baseline standards (99 containers, Day 1, real CPU %, verification scripts, frozen reference).

#### Generated Outputs

| Output | Location |
|--------|----------|
| Primary objective | §2.4 of this document |
| Research questions table | §2.4 of this document |
| Success criteria checklist | §2.6 of this document |

#### Conclusions

Task 4 is complete. Objectives and success criteria provide measurable exit conditions for the full study (Phases 1–6).

#### Next Task

**Task 5 — Define Explicit Non-Goals and Implementation Philosophy.**

---

### Task 5 — Define Explicit Non-Goals and Implementation Philosophy

**Status:** Complete  
**Date:** 2026-07-17

#### Objective

Document what is **explicitly out of scope** during Global GRU baseline establishment and define the implementation philosophy that prevents scope creep during Phases 4–6.

#### Methodology

1. Listed deferred work items (Peak-Aware, preprocessing changes, tuning, ablations, unseen containers).
2. Defined implementation prohibitions for the baseline experiment.
3. Recorded that future enhancements are separate experiments after baseline freeze.

#### Implementation Philosophy (Locked)

During the Global GRU baseline experiment:

| Prohibited | Reason |
|------------|--------|
| Peak-Aware Learning | Separate study track after freeze |
| Model-structure optimisation | Would change locked specification |
| Hyperparameter tuning | Notebook values locked as-is |
| Additional features | Would change experimental configuration |
| Preprocessing changes | Shared pipeline must remain identical |

**Objective:** Establish a scientifically valid, reproducible baseline — not maximised accuracy.

#### Generated Outputs

| Output | Location |
|--------|----------|
| Non-goals table | §2.7 of this document; [README](README.md) §1.7 |
| Implementation philosophy | [README](README.md) §1.5; [Phase 0.5](phase-03-5-baseline-specification.md) §4.5.4 |

#### Conclusions

Task 5 is complete. Scope boundaries and implementation philosophy are locked for all subsequent phases.

#### Next Task

**Task 6 — Design Phased Research Workflow.**

---

### Task 6 — Design Phased Research Workflow

**Status:** Complete  
**Date:** 2026-07-17

#### Objective

Design the complete phased workflow for the Global GRU baseline study, including the Phase 0.5 specification gate before any implementation code is written.

#### Methodology

1. Aligned phase structure with Peak-Aware Hybrid research log format.
2. Inserted Phase 0.5 (Baseline Specification) as pre-implementation gate — mirrors Peak-Aware Phase 3 design lock before Phase 4 implementation.
3. Mapped each phase to a single primary deliverable.
4. Documented dependency order: Phase 1 → 2 → 3 → 0.5 (approve) → 4 → 5 → 6.

#### Implementation Decisions

| Decision | Rationale |
|----------|-----------|
| Phase 0.5 before Phase 4 | Formal research contract before coding — prevents scope drift |
| Separate evaluation phase (Phase 5) | Matches Hybrid baseline verification pattern |
| Phase 6 = freeze + methodology comparison | Completes Research Objective 2 |
| No Phase 7 | Baseline study ends at frozen reference + comparison |

#### Generated Outputs

| Output | Location |
|--------|----------|
| Phased workflow table | §2.8 of this document |
| README workflow index | [README](README.md) §1.3 |
| Phase documents 2–6 (planned sections) | `docs/global-gru-baseline/phase-*.md` |

#### Conclusions

Task 6 is complete. The seven-step workflow (Phases 1, 2, 3, 0.5, 4, 5, 6) is locked.

#### Next Task

**Task 7 — Create Research Documentation Folder and Phase Index.**

---

### Task 7 — Create Research Documentation Folder and Phase Index

**Status:** Complete  
**Date:** 2026-07-17

#### Objective

Create the `docs/global-gru-baseline/` research log with README index, phase documents, and cross-links — matching the structure of `docs/peak-aware-hybrid/`.

#### Methodology

1. Created `docs/global-gru-baseline/README.md` as primary index and research overview.
2. Created phase documents:
   - `phase-01-research-planning.md` (this document)
   - `phase-02-methodology-audit.md`
   - `phase-03-pipeline-design.md`
   - `phase-03-5-baseline-specification.md`
   - `phase-04-implementation.md`
   - `phase-05-evaluation.md`
   - `phase-06-baseline-freeze-and-comparison.md`
3. Updated `docs/README.md` with link to Global GRU baseline study.
4. Added navigation links, status tables, and appendices (frozen files, related docs).

#### Generated Outputs

```
docs/global-gru-baseline/
├── README.md
├── phase-01-research-planning.md
├── phase-02-methodology-audit.md
├── phase-03-pipeline-design.md
├── phase-03-5-baseline-specification.md
├── phase-04-implementation.md
├── phase-05-evaluation.md
└── phase-06-baseline-freeze-and-comparison.md
```

| Output | Location |
|--------|----------|
| Research log index | [README.md](README.md) |
| Project docs link | [docs/README.md](../README.md) |

#### Conclusions

Task 7 is complete. The Global GRU baseline research log is established and linked from the project documentation index.

#### Next Task

**Task 8 — Lock Phase 1 Research Planning Record.**

---

### Task 8 — Lock Phase 1 Research Planning Record

**Status:** Complete  
**Date:** 2026-07-17

#### Objective

Finalise and lock the Phase 1 research planning record. Confirm all planning tasks are complete and document the handoff to Phase 2.

#### Methodology

1. Verified all Tasks 1–7 deliverables exist and are cross-linked.
2. Confirmed Phase 1 exit criteria (§2.11) are satisfied.
3. Recorded Phase 1 completion status in README index.
4. Defined next phase entry point: Phase 2 Methodology Audit (already completed in same documentation pass).

#### Phase 1 Summary

| Item | Status |
|------|--------|
| Research context reviewed | ✓ |
| Global GRU prototype audited | ✓ |
| Comparison philosophy documented | ✓ |
| Objectives and success criteria defined | ✓ |
| Non-goals and implementation philosophy locked | ✓ |
| Phased workflow designed | ✓ |
| Research log created | ✓ |
| Phase 1 record locked | ✓ |

#### Conclusions

**Phase 1 is complete.** All eight tasks finished. The Global GRU baseline study has a defined scope, comparison philosophy, success criteria, and phased workflow.

**Handoff:** Phase 2 (Methodology Audit) and Phase 3 (Pipeline Design) were completed in the same documentation pass. Phase 0.5 (Baseline Specification) is documented and awaits approval before Phase 4 Implementation begins.

#### Next Phase

**Phase 2 — Methodology Audit & Fair Comparison:** [phase-02-methodology-audit.md](phase-02-methodology-audit.md)  
**Immediate gate before coding:** [Phase 0.5 — Baseline Specification](phase-03-5-baseline-specification.md)

---

## 2.12 Phase 1 Status

**Phase 1 is complete.**  
**Date locked:** 2026-07-17  
**Next actionable gate:** Approve [Phase 0.5 — Baseline Specification](phase-03-5-baseline-specification.md), then begin [Phase 4 — Implementation](phase-04-implementation.md).
