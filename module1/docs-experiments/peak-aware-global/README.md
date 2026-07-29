# Peak-Aware Global GRU — Research Documentation

**Project:** DracaSys Module 1 — Long-Term CPU Utilization Forecasting  
**Institution:** University of Moratuwa — Bachelor of Information Technology (Final Year Research Project)

**Thesis framing:** One Peak-Aware learning methodology applied to two forecasting architectures. This folder documents the **Global GRU application** of that shared methodology. It is **not** a second independent Peak-Aware research project.

---

## Methodology Tree

```
Peak-Aware Learning (thesis methodology)
│
├── Hybrid Prophet + GRU
│      ├── Control:  experiments/baseline_reference_2026-07-14/
│      └── Treatment: experiments/peak_aware_2026-07-14_164518/  [complete]
│
└── Global GRU
       ├── Control:  experiments/global_gru_baseline_2026-07-17_121748/  [official]
       │              Superseded: experiments/global_gru_reference_2026-07-17/  [historical, patience=3]
       └── Treatment: experiments/peak_aware_global_validation_vs1_2026-07-17_170939/  [official]
              └── Archived: experiments/peak_aware_global_2026-07-17_164737/  [Historical – optimization-limited, patience=3]
```

---

## Documentation Index

| Stage | Document | Status |
|-------|----------|--------|
| 1 | [Design Lock](01-design-lock.md) | **Complete** |
| 2 | [Implementation Plan](stage-02-implementation-plan.md) · [Implementation Record](02-implementation-record.md) | **Complete** |
| 3 | [Evaluation Plan](stage-03-evaluation-plan.md) · [VS-1 Training Dynamics Diagnostic](stage-03-training-dynamics-diagnostic-plan.md) · [Evaluation Record](03-evaluation-record.md) | **Complete** |
| 4 | [Discussion Record](04-discussion-record.md) | **Complete** |

**Design lock artifact:** `experiments/peak_aware_global_design_2026-07-17/design_lock.json`  
**Official treatment run:** `experiments/peak_aware_global_validation_vs1_2026-07-17_170939/`  
**Stage 2 implementation lock (historical run):** `experiments/peak_aware_global_2026-07-17_164737/phase2_implementation.json`  
**Official promotion record:** `experiments/peak_aware_global_validation_vs1_2026-07-17_170939/official_treatment_promotion.json`

---

## Authority & Cross-References

| Resource | Role |
|----------|------|
| [Peak-Aware Hybrid](../peak-aware-hybrid/README.md) | **Methodology authority** — peak definition, λ, loss, evaluation protocol (Phases 1–6 complete) |
| [Global GRU Baseline](../global-gru-baseline/README.md) | Frozen control context — `global_gru_v1` specification |
| [Phase 0.5 Baseline Spec](../global-gru-baseline/phase-03-5-baseline-specification.md) | Future Peak-Aware single-variable contract |
| `experiments/global_gru_baseline_2026-07-17_121748/` | **Official Global control** for this study (supersedes `global_gru_reference_2026-07-17/`) |
| `experiments/peak_aware_2026-07-14_164518/` | Locked Hybrid Peak-Aware treatment (methodology reference) |

---

## Study Stages

| Stage | Name | Status |
|-------|------|--------|
| 1 | Design Lock (including methodology inheritance) | **Complete** |
| 2 | Implementation + Fairness Verification | **Complete** |
| 3 | Evaluation + Verification | **Complete** |
| 4 | Results, Discussion & Architecture Sensitivity Synthesis | **Complete** |

---

## Primary vs Secondary Comparisons

| Comparison | Role |
|------------|------|
| Global GRU vs Peak-Aware Global GRU | **Primary** — answers Global-track research questions |
| Peak vs non-peak (within Global pair) | **Primary** — Tier 2 contribution metrics |
| Hybrid vs Peak-Aware Hybrid | **Already locked** — cite only |
| Hybrid vs Global (methodology) | **Already locked** — cite only |
| Architecture Sensitivity (Δ Hybrid vs Δ Global) | **Secondary synthesis** — Stage 4 §4.7 only |

---

## Fair Comparison Principle

The official Global GRU baseline (`experiments/global_gru_baseline_2026-07-17_121748/`) must not be modified. Peak-Aware Global work uses parallel modules and separate experiment outputs only. The only intended difference between control and treatment is the GRU training objective (timestep-weighted MSE with λ = 5).

**Mandatory sequence fairness rule (Stage 2 gate):** Peak-Aware Global must produce **identical** `X_all`, `y_all`, and sample ordering as the frozen baseline. Only `W_all` (the weight matrix) may differ. Fairness verification requires **10 checks PASS** on the official treatment (see [Stage 2 plan](stage-02-implementation-plan.md)).

---

## Official Treatment Promotion (2026-07-17)

The original primary run (`_164737`, EarlyStopping patience=3) is archived as **Historical – optimization-limited**. VS-1 (`_170939`, patience=10) is the **official Peak-Aware Global treatment** following the [VS-1 validation study](stage-03-training-dynamics-diagnostic-plan.md). Stage 1 design lock (λ, P90, architecture, evaluation protocol) is unchanged; only the training stopping patience was amended for the official model.

---

## Frozen Files (Do Not Modify)

- `utils/global_config.py`
- `utils/global_training.py`
- `utils/global_inference.py`
- `utils/global_evaluation.py`
- `utils/global_artifacts.py`
- `utils/global_plotting.py`
- `utils/sequence_utils.py`
- `notebooks/global_gru_model.ipynb`
- `models/global_gru.keras`
- `models/global_gru_metadata.json`
- `experiments/global_gru_baseline_2026-07-17_121748/` (official baseline — do not modify)
- `experiments/global_gru_reference_2026-07-17/` (superseded historical reference)

Shared peak utilities (`utils/peak_detection.py`) may be **extended** (e.g. `align_with_longterm_sequences()`) but existing Hybrid-proven functions must remain backward-compatible.

---

*Last updated: 2026-07-17 — Stage 4 discussion locked; Peak-Aware Global GRU study complete*
