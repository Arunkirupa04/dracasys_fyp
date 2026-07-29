# CSRLE Validation Protocol

Pre-GRU gates for **Protocol v2 (boundary-safe)**.  
Authoritative v2 results: `experiments/synthetic_residual_learnability_2026-07-25_164307/synthetic_validation/`

v1 results (failed clip gate, preserved): `experiments/synthetic_residual_learnability_2026-07-25_163013/`

Thresholds are **pre-specified experimental criteria**, not universal statistical standards.

---

## v2 Pre-GRU gate summary

| Gate | B0 | B1 | C0 | C1 | Control A |
|------|:--:|:--:|:--:|:--:|:---------:|
| Temporal generator | PASS | PASS | PASS | PASS | — |
| Amplitude preservation | PASS | PASS | — | — | — |
| Clip/boundary | PASS | PASS | PASS | PASS | — |
| Prophet retention | PASS | PASS | — | — | Prophet reconciled |
| Permitted for GRU | Yes | Yes | Yes | Yes | Yes |

**Frozen config written:** Yes (all gates passed)

---

## 1. Boundary-safety (α_c) — NEW in v2

| Field | Value |
|-------|-------|
| **Purpose** | Prevent waveform distortion from hard clipping |
| **Method** | α_c = min(1, min_t α_t^max) on train N_target only |
| **Threshold** | Clip gate after α scaling (see §6) |
| **Rationale** | v1 showed ~50% clip on near-idle containers |
| **B0 result** | 90.9% α=1; 8 containers α=0; mean α=0.914 |
| **Pass/Fail** | PASS (downstream clip gate) |

Artifact: `alpha_distribution_summary.json`, `alpha_map_*.csv`

---

## 2. Amplitude-preservation gate — NEW in v2

| Check | Threshold | B0 | B1 |
|-------|-----------|----|----|
| fraction α=1 | ≥ 85% | 90.9% ✓ | 86.9% ✓ |
| fraction α=0 | ≤ 10% | 8.1% ✓ | 9.1% ✓ |
| fraction α<0.25 | ≤ 10% | 8.1% ✓ | 9.1% ✓ |
| median α | ≥ 0.95 | 1.0 ✓ | 1.0 ✓ |
| mean κ_eff | ≥ 0.08 | 0.091 ✓ | 0.090 ✓ |

**Pass/Fail:** PASS both B0 and B1

---

## 3. B0 generator (v2 rerun)

| Check | Threshold | Observed | Pass |
|-------|-----------|----------|------|
| ACF1(z) mean | ≥ 0.30 | 0.924 | ✓ |
| Rollout RMSE | ≤ 1e−6 | 0.0 | ✓ |
| κ_eff match | rel ≤ 0.02 | ~0 | ✓ |
| Clip fail fraction | ≤ 5% | **0%** | ✓ |
| Cohort mean clip | — | 0.000013 | ✓ |
| Worst clip | — | 0.13% (c_11087) | ✓ |

**Overall: PASS**

---

## 4. B1 generator (v2 rerun)

| Check | Observed | Pass |
|-------|----------|------|
| ACF1 mean | 0.477 | ✓ |
| Clip fail fraction | 0% | ✓ |
| Worst clip | 0.26% (c_15640) | ✓ |
| Amplitude | PASS | ✓ |

**Overall: PASS**

---

## 5. C0 / C1 (v2 rerun)

| Check | C0 | C1 |
|-------|----|----|
| \|ACF1\| mean | 0.030 ✓ | 0.028 ✓ |
| Effective std match | PASS | PASS |
| Clip fail fraction | 0% | 0% |

**Overall: PASS both**

---

## 6. Clip/boundary gate (v2)

| Condition | Fail fraction (>1% clip) | Worst rate |
|-----------|--------------------------|------------|
| B0 | 0% | 0.13% |
| B1 | 0% | 0.26% |
| C0 | 0% | 0% |
| C1 | 0% | 0% |

Hard clip remains as physical safeguard; α scaling makes clipping rare.

---

## 7. Prophet retention — B0 (v2 rerun)

| Metric | v1 (163013) | v2 (164307) | Gate |
|--------|-------------|-------------|------|
| ρ(N,I) | 0.999 | 0.999 | ✓ |
| VS | 0.002 | 0.002 | ✓ |
| VR | 1.000 | 1.000 | ✓ |
| ~4h period | 3.93 h | 3.93 h | diagnostic |
| Identity max err | ~1e−14 | ~1e−14 | ✓ |

**Prophet absorption:** VS ≈ 0.002 — structure **not** absorbed. **PASS**

---

## 8. Prophet retention — B1 (v2 rerun)

| Metric | v2 | Gate |
|--------|-----|------|
| ρ(N,I) | 0.985 | ✓ |
| VS | 0.032 | ✓ |
| VR | 1.033 | ✓ |

**PASS**

---

## 9. Condition A Prophet reconciliation

| Scope | Mean MAE | Authoritative for |
|-------|----------|-------------------|
| Day-1 (96 steps) | **1.733** | CSRLE / Hybrid Day-1 evaluation |
| Full validation (~154 steps) | **1.847** | Residual-pattern analysis §3 |
| Reference documented | 1.8474 | `docs/residual-pattern-analysis-findings.md` |

**Explanation:** Same 99 containers, same Prophet config, same real CPU % — **different evaluation horizons**. Not a methodology bug.

Artifact: `control_prophet_mae_reconciliation.csv`

Hybrid reproduction gate: **PASS** (Stage 1, see §11)

---

## 11. Condition A Hybrid reproduction gate (Stage 1)

| Field | Value |
|-------|-------|
| **Purpose** | CSRLE implementation reproduces frozen Hybrid methodology |
| **Frozen eval path sanity** | Frozen model + CSRLE eval path vs `evaluation_df.csv` |
| **Sanity result** | PASS (max diff ≈ 10⁻¹⁶) |
| **Retrain comparison** | New Condition A GRU vs frozen baseline |
| **Tolerances** | \|ΔMAE mean\| ≤ 0.10; \|ΔRMSE mean\| ≤ 0.15; rank-corr ≥ 0.95 |
| **CSRLE mean MAE / RMSE** | 1.745 / 2.386 |
| **Frozen mean MAE / RMSE** | 1.746 / 2.388 |
| **Rank correlation** | 0.99998 |
| **Pass/Fail** | **PASS** |

Artifact: `stage1_control_reproduction/evaluation/reproduction_summary.json`

Synthetic GRU (B0/B1/C0/C1): **not started** — awaiting approval.

---

## 10. Synthetic config freeze (v2)

| Field | Value |
|-------|-------|
| **Condition** | All generator + retention gates pass |
| **Result** | **FROZEN** |
| **Path** | `experiments/..._164307/config/synthetic_config_frozen.json` |

---

## Permitted GRU conditions (post Stage 1)

Pre-GRU permitted: all five conditions (frozen config written).

Stage 1 trained: **control only**.

Synthetic conditions (B0/B1/C0/C1): **awaiting explicit approval**.
