# CSRLE Implementation Log

---

## 2026-07-25 — Protocol Amendment 1 (v2 boundary-safe)

| Field | Detail |
|-------|--------|
| **Task** | Implement train-derived α_c; rerun pre-GRU validation |
| **Approval** | Supervisor-approved amendment before any GRU training |
| **Files created** | `utils/csrle/boundary.py`, `docs/csrle/protocol_amendments.md` |
| **Files modified** | `utils/csrle/{config,dataset,validation_gates,diagnostics}.py`, `scripts/run_csrle_pre_gru_validation.py`, CSRLE docs |
| **v1 preserved** | `experiments/synthetic_residual_learnability_2026-07-25_163013/` — not modified |
| **α rule** | Strict train min including zero margins; see protocol_amendments.md |
| **C0/C1 fix** | z_train std normalization for exact effective-std match |
| **Result run** | `experiments/synthetic_residual_learnability_2026-07-25_164307/` |
| **Decision** | All pre-GRU gates passed; frozen config written; await GRU approval |

### Run 164307 results

| Stage | Outcome |
|-------|---------|
| B0 generator + amplitude + clip | PASS |
| B1 generator + amplitude + clip | PASS |
| C0/C1 generator + clip + std match | PASS |
| B0 retention (VS≈0.002, ~3.93h period) | PASS |
| B1 retention | PASS |
| Prophet MAE reconciliation | Day-1 1.733; full val 1.847 |
| Frozen config | Written |
| Permitted GRU | control, b0_lc, b1_snar, c0_iid, c1_iid |
| GRU training | **NOT performed** |

---

## 2026-07-25 — v1 pre-GRU (163013)

| Field | Detail |
|-------|--------|
| **Outcome** | Clip gate FAILED; retention PASSED; no frozen config |
| **Action** | Led to Amendment 1 |

---

## Pending

- [x] Hybrid control reproduction gate (Stage 1 — PASS)
- [x] Synthetic-condition GRU training (Stage 2 — COMPLETE)
- [x] Full CSRLE evaluation + results.md / discussion.md

---

## 2026-07-25 — Stage 2 Synthetic GRU training and evaluation

| Field | Detail |
|-------|--------|
| **Task** | Train independent GRUs for B0/B1/C0/C1; full evaluation suite |
| **Script** | `scripts/run_csrle_stage2_synthetic_gru.py` |
| **Modules** | `utils/csrle/stage2_{training,baselines,metrics,stats,plots}.py` |
| **Artifacts** | `experiments/..._164307/stage2_synthetic_gru/` |
| **Runtime** | ~26 min (4× train + 4× 99-container Prophet+eval) |
| **B0 vs C0** | r +0.035 [0.004, 0.067]; std ratio +0.030; R² still negative |
| **B1 vs C1** | r +0.030 [0.004, 0.057]; weaker than B0/C0 |
| **Ridge vs GRU (B0)** | Ridge r 0.158 > GRU 0.084 |
| **CPU translation** | B0 Hybrid −Prophet MAE −0.006; others ~flat |
| **Condition A** | Not retrained; Stage 1 reference used |
| **Decision** | Stage 2 complete; see `docs/csrle/stage2_report.md` |

---

## 2026-07-25 — Stage 1 Condition A Hybrid reproduction

| Field | Detail |
|-------|--------|
| **Task** | Train new Condition A GRU; compare to frozen baseline |
| **Script** | `scripts/run_csrle_stage1_control_reproduction.py` |
| **Artifacts** | `experiments/..._164307/stage1_control_reproduction/` |
| **Frozen eval path sanity** | PASS (max MAE diff ≈ 4.4×10⁻¹⁶) |
| **Reproduction gate** | PASS (ΔMAE mean −0.001, rank-corr 0.99998) |
| **New model MAE/RMSE** | 1.745 / 2.386 vs frozen 1.746 / 2.388 |
| **B0/B1/C0/C1** | NOT trained |
| **Decision** | READY FOR SYNTHETIC GRU STAGE (pending approval) |
