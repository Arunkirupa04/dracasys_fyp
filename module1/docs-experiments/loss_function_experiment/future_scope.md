# Future Scope

Explicitly **out of scope** for LFHE v1. Listed for thesis limitations and follow-on research — **not** to be added during initial implementation without new protocol amendment.

---

## Loss function extensions

| Follow-up | Why deferred |
|-----------|--------------|
| **λ sensitivity sweep** (0.1, 0.5, 1, 2, 5) | Constitutes hyperparameter tuning — violates hypothesis-test framing of LFHE v1. Requires **new protocol (LFHE v2)** if pursued. |
| **MSE + λ(1 − r) joint loss** | Second competing objective; λ balance ambiguous |
| **Symmetric log-variance penalty** (penalize over-dispersion too) | Not observed failure mode; adds complexity |
| **Per-horizon dispersion penalty** | Multiple hypotheses; beyond single-factor design |
| **Batch-pooled σ̂ vs per-sequence σ̂ ablation** | Methodological variant — only if DA-MSE inconclusive |

### λ sensitivity analysis (explicitly deferred)

LFHE v1 freezes **λ = 1.0** before any training. Rationale:

- The experiment tests whether **introducing dispersion-aware supervision** (at equal nominal weight to O(1) MSE-scale terms) alters variance collapse — not which λ maximizes performance.
- Searching for an optimal λ adds a **second experimental factor**, invites data snooping on B0, and weakens the causal claim that objective design (not λ magnitude) matters.
- A null result at λ = 1.0 is interpreted as **failure to reject H₀ at this supervision strength**, not as motivation to tune λ within v1.

**Permitted follow-up:** LFHE v2 pre-registering a **small, fixed set** of λ values as a **separate** multi-arm study — not a post-hoc sweep on v1 outputs.

---

## Architecture and model changes

| Follow-up | Why deferred |
|-----------|--------------|
| Learned variance head (Gaussian NLL) | Architecture change — confounds loss hypothesis |
| Linear residual head / AR baseline inside Hybrid | Different inductive bias — Ridge already covers |
| Deeper/wider GRU | Capacity study, not objective study |
| Attention / Transformer residual learner | New architecture benchmark |
| Replace GRU with Ridge in production Hybrid | Not supported by CSRLE alone |

---

## Data and protocol extensions

| Follow-up | Why deferred |
|-----------|--------------|
| B1, C0, C1 LFHE arms | CSRLE generator questions already answered in Stage 2 |
| New synthetic generators | CSRLE v2 frozen |
| Real-data-only experiment without B0 | Weaker falsifiability without positive control |
| Multi-cloud datasets | Out of FYP scope |
| Peak-aware + DA-MSE combination | Confounds two objective interventions |

---

## Evaluation extensions

| Follow-up | Why deferred |
|-----------|--------------|
| Day-2 recursive extension | Secondary to Day-1 hypothesis |
| Peak-subset metrics as primary | Peak-aware already limited; dispersion is core |
| Spectral loss as training objective | Ridge analysis showed spectral parity — low priority |
| Calibration / CRPS probabilistic metrics | Requires distributional predictions |

---

## If H₁ is supported

Permitted **future** work (new protocol each):

1. **LFHE v2 — Condition A primary** with DA-MSE if B0 supports H₁.
2. **LFHE v2 — λ frozen from theory sensitivity** (single pre-registration, not sweep).
3. **Hybrid production experiment** — DA-MSE in full Hybrid retrain (separate from CSRLE).

## If H₁ is rejected

Permitted **future** work:

1. **Architecture ablation** — linear skip connection, smaller GRU, AR features in input.
2. **Optimizer study** — learning rate, gradient clipping (not loss).
3. **Target engineering** — predict differenced residuals, multi-step horizon factorization.

## If inconclusive

1. Diagnose per-sequence σ_y floor cases.
2. Consider **two-sided** dispersion penalty variant under new protocol.
3. Re-examine early stopping on shape metrics vs loss metrics.

---

## Thesis positioning

LFHE v1 is intended as a **single decisive chapter section** — one prospective test, one alternative loss, one primary dataset (B0). Breadth is intentionally sacrificed for **causal clarity**.

Do not expand into a loss-function survey paper within this FYP timeline.
