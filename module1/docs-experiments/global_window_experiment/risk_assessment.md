# Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Sequence reduction** | Certain (by design) | Medium | Pre-computed counts; G384 excluded; monitor min sequences/container |
| **Overfitting** | Medium (G288) | High | Early stopping unchanged; compare val_loss vs test MAE; regularisation frozen |
| **No forecast gain despite TMA memory** | Medium–High | Medium | Pre-register as valid null result; cite RLLA/HCERL precedents |
| **Compute / time increase** | Certain | Low | ~2× wall time for G288 acceptable; sequential variant training |
| **G96 replication drift** | Medium | High | G96 replication gate vs frozen baseline before interpreting variants |
| **Dataset too short for G384** | High | High | **Exclude G384** from primary arms |
| **Subgroup underpowering** | Medium | Medium | Tertiles ≈ 33 containers each — report effect sizes, not only p-values |
| **Inference window mismatch** | Low | Critical | Protocol explicitly updates inference to `input_window` steps |
| **Accidental baseline modification** | Low | Critical | Isolated `utils/ggtce/` module; no edits to frozen Global baseline dir |
| **Confounding from seed/variance** | Low | Medium | Fixed seed 42; same as Global baseline |
| **Horizon non-stationarity** | Medium | Low | Day-1 band analysis; validation only ~1.5 days |

## Diminishing returns risk (G288)

G288 exposes +13.5 pp memory vs G192 but **−46% sequences**. Risk that marginal memory gain does not compensate for reduced training data — **pre-specified** as key research question (H3).

## Thesis narrative risk

If GGTCE null: thesis still strengthened — *"TMA showed long memory; empirical Global GRU test showed it is not exploitable under frozen architecture."*

If GGTCE positive: thesis gains adaptive-context motivation for Global path only.

## Go / no-go

| Factor | Status |
|--------|--------|
| Scientific justification (TMA) | ✅ Strong |
| Sufficient sequences (G96–G288) | ✅ Yes |
| Dataset length adequate | ✅ Yes for G96–G288; ⚠️ marginal for G384 |
| Compute practical | ✅ Yes |
| Isolation feasible | ✅ Yes |

**Overall risk level:** **Acceptable** for G96 + G192 + G288.
