# Prophet Retention Analysis (CSRLE)

## Motivation

Before training the GRU, we must verify that injected synthetic structure **N(t)** survives Prophet fitting in validation residuals **I(t) = N(t) − ΔP(t)**. If Prophet absorbs structure into ŷ_B^P, the GRU never sees it in R_B — invalidating the positive-control test.

## Invalid formulation (rejected)

Using the **same** Prophet model on y_base and y_syn:

```
ΔP = P_B(y_syn) − P_B(y_base) ≡ 0   (same fitted model, same timestamps)
```

This is **algebraically invalid** for cross-condition retention and was explicitly rejected in the design correction.

## Valid diagnostic (implemented)

Fit **separate** Prophet models on Condition A and Condition B **train** data:

| Symbol | Definition |
|--------|------------|
| **P_A** | Prophet fitted on Condition A (control) train |
| **P_B** | Prophet fitted on Condition B (synthetic) train |
| **ŷ_A^P(t)** | P_A val Day-1 forecast (real CPU %) |
| **ŷ_B^P(t)** | P_B val Day-1 forecast (real CPU %) |
| **y_A(t)** | Control val CPU % |
| **y_B(t)** | Synthetic val CPU % (post-clip) |
| **N(t)** | y_B(t) − y_A(t) effective injection |
| **ΔP(t)** | ŷ_B^P(t) − ŷ_A^P(t) |
| **I(t)** | N(t) − ΔP(t) injected structure retained in residual space |
| **R_A(t)** | y_A(t) − ŷ_A^P(t) |
| **R_B(t)** | y_B(t) − ŷ_B^P(t) |

### Identity

```
I(t) = N(t) − ΔP(t)
     = (y_B − y_A) − (ŷ_B^P − ŷ_A^P)
     = (y_B − ŷ_B^P) − (y_A − ŷ_A^P)
     = R_B(t) − R_A(t)
```

**Pre-GRU verification:** max |I − (R_B − R_A)| ≈ 1.4×10⁻¹⁴ (machine epsilon).  
Requires **post-clip** N = y_syn − y_base in manifest.

## What the decomposition measures

- **N:** Ground-truth injected deviation from real CPU on val
- **ΔP:** How much Prophet's forecast shifts between conditions
- **I:** Residual-domain structure available after Prophet — the relevant target for GRU
- **VS = Var(ΔP)/Var(N):** Fraction of injection variance absorbed by Prophet (low → good for GRU test)
- **VR = Var(I)/Var(N):** Fraction retained (high → structure in R_B − R_A)
- **ρ(N,I):** Linear alignment of injection with retained structure

This is **diagnostic**, not causal attribution: it quantifies co-movement under the controlled design, not Prophet's internal mechanism.

## B0 periodicity and Prophet absorption (pre-GRU findings)

**Authoritative:** `experiments/synthetic_residual_learnability_2026-07-25_163013/synthetic_validation/prophet_retention_report.json`

### B0-LC

| Metric | Value | Interpretation |
|--------|-------|----------------|
| ρ(N,I) mean | 0.999 | Nearly all injection retained in I |
| VS mean | 0.002 | Prophet absorbed ~0.2% of injection variance |
| VR mean | 1.000 | Full variance in residual domain |
| ρ(N,ΔP) mean | 0.018 | ΔP largely orthogonal to N |
| Dominant N period (pooled val FFT) | ~3.93 h | Consistent with ~4 h design target |
| ACF(R_B) lags 1–10 mean | 0.191 | Temporal structure remains in R_B |

**Conclusion:** Under the pre-GRU retention gate, Prophet **did not** absorb the B0 limit-cycle structure. B0 would be a suitable positive control **on retention grounds**.

**However:** B0 **failed the generator clip gate** (see `validation_protocol.md`), so B0 GRU is blocked under the full pre-specified protocol—not because of Prophet absorption.

### B1-SNAR

| Metric | Value |
|--------|-------|
| ρ(N,I) mean | 0.985 |
| VS mean | 0.032 |
| VR mean | 1.033 |

Prophet retains most B1 structure in residuals. B1 also failed clip gate.

## Retention gates (pre-specified)

See `utils/csrle/config.py` — `B0_RETENTION_GATES`, `B1_RETENTION_GATES`.

Both **PASSED** in run `2026-07-25_163013`.

## Per-container artifacts

- `synthetic_validation/prophet_retention_b0.csv`
- `synthetic_validation/prophet_retention_b1.csv`
- `synthetic_validation/prophet_retention_report.json`

Implementation: `utils/csrle/prophet_retention.py`
