# MDC vNext — Issues Register & Improvement Plan

**Status:** Active (post `run-final` research audit)  
**Date:** 2026-07-13  
**Audit sources:**  
`notebook/run-final/mdc_preprocess_vNext_mc_final.ipynb`,  
`notebook/run-final/mdc_model_vNext_final.ipynb`,  
`notebook/run-final/mdc_drift_aware_final.ipynb`  
**Related:** [`claim_evidence_matrix.md`](claim_evidence_matrix.md), [`thesis_results_discussion.md`](thesis_results_discussion.md)

---

## 1. Executive summary

| Item | Value |
|------|-------|
| Thesis readiness verdict | **B — Acceptable but requires improvements** |
| Overall readiness score | **~65 / 100** |
| Evidence freeze for claims | Prefer **`run-final`** notebook outputs over older locked 0.7402 figures until reconciled |
| Adaptive / fine-tune success claims | **Do not claim** unless a new freeze shows `claim_ok=True` |

**Bottom line:** The pipeline is implementation-complete and suitable as a final-year project *after* metric lock, claim cleanup, and a short list of experimental gaps are closed.

---

## 2. Ground-truth metrics (`run-final` freeze)

Use these numbers until docs are re-locked or a new run restores the previous lock.

| Config | ROC-AUC | F1 | MCC | Notes |
|--------|---------|-----|-----|-------|
| vNext default | **0.7042** | 0.6935 | 0.4892 | Methods reference for this freeze |
| Multi-seed mean ± std | **0.7529 ± 0.023** | 0.7341 ± 0.017 | 0.5771 ± 0.041 | Abstract robustness |
| HPO best | **0.8138** | 0.7736 | 0.6260 | Best-case only |
| Isolation Forest | **0.6741** | 0.7104 | 0.5034 | Classical baseline |
| Exp A (lineage) | 0.7163 | — | — | Prior reference |
| Offline thr (raw) | **≈ −0.1775** | — | — | From this freeze’s `metrics_vnext.json` |
| Stream mode | `container_lexsort` | — | — | Not timestamp |
| Integrity gates | **PASS** | FPR 0.2345 | — | Fixed path only |
| Adaptive `claim_ok` | **False** | F1/recall → 0 | — | Do not claim improvement |

**Doc conflict:** `claim_evidence_matrix.md` still locks default ROC **0.7402**, thr **≈ −0.1434**, F1 **≈ 0.741**. That must be reconciled (see ISS-01).

---

## 3. Issues register

Severity: **Critical** / **High** / **Medium** / **Low**

### 3.1 Critical

| ID | Issue | Evidence | Impact if ignored |
|----|-------|----------|-------------------|
| **ISS-01** | Headline metric lock ≠ `run-final` | Docs: 0.7402 / thr −0.1434; run-final: **0.7042** / thr **−0.1775** | Wrong Abstract/Results; viva failure risk |
| **ISS-02** | Adaptive success claim unsupported | Last-half adaptive FPR/F1/recall = 0; `claim_ok=False` | Invalid research claim if stated as success |
| **ISS-03** | v1 “FPR↓ ~0.92” must never be cited | Historical double-invert bug | Scientific invalidation |

### 3.2 High

| ID | Issue | Evidence | Impact if ignored |
|----|-------|----------|-------------------|
| **ISS-04** | Default scoring path ≠ multi-seed/HPO path | Default uses `feat_std`; multi-seed/HPO use `feat_std=None` + flip | Inflated cross-experiment comparison |
| **ISS-05** | Default below Exp A in this freeze | 0.7042 vs Exp A 0.7163 | Undermines “improved over Exp A” unless multi-seed/HPO framed carefully |
| **ISS-06** | Timestamp streaming inactive | Drift stream = `container_lexsort`; MC npz has `ts_*` but drift loads binary `windows_vnext.npz` | Weakens temporal drift narrative |
| **ISS-07** | Val/test may share sessions | Holdout split is window-level 50/50, not session-disjoint | Optimistic F1-optimal threshold |
| **ISS-08** | Ablations coded but not run | `RUN_ABLATIONS=False` in model final | Claim matrix C11 still pending |
| **ISS-09** | Thin baselines | Isolation Forest only; no OCSVM / LSTM AE / PCA AE | Weak comparative Results section |
| **ISS-10** | Fine-tune result unstable vs thesis draft | run-final ΔAUC **+0.0319**; draft cites **≈ −0.06** | Adaptation story not frozen |

### 3.3 Medium

| ID | Issue | Evidence | Impact if ignored |
|----|-------|----------|-------------------|
| **ISS-11** | Ensemble implemented but unused | Ensemble test AUC 0.6652 < AUC-best 0.7042 | Fine to omit; state “rejected on val” |
| **ISS-12** | Per-attack sparsity | Only CVE-2020-13379 + Node-RED Recon have useful n; others n≤19 | Overclaiming uniform attack coverage |
| **ISS-13** | Very low TPR at strict FPR | TPR @ FPR=5% ≈ 0.0041 | Operating-point honesty needed |
| **ISS-14** | Latency numbers differ | Doc ~3.36 ms; this run ~1.53 ms | Hardware-specific; cite profile |
| **ISS-15** | Drive/FUSE sync unreliable | UI empty after “Copied OK”; zip fallback used | Engineering friction, not science |
| **ISS-16** | `mdc_dataloader.ipynb` is legacy | Not part of vNext final path | Confusion if cited as current |

### 3.4 Low

| ID | Issue | Evidence | Impact if ignored |
|----|-------|----------|-------------------|
| **ISS-17** | No live deployment | Simulated stream only | Acceptable as limitation |
| **ISS-18** | Single dataset | MDC only | Acceptable if Future Work states multi-dataset |
| **ISS-19** | No explainability module | Not implemented | Optional for FYP |
| **ISS-20** | No detection-delay metric | Not measured | Optional enhancement |

---

## 4. Preprocess issues (detail)

| Check | Result | Notes |
|-------|--------|-------|
| Window / stride | OK | T=10 (150s), stride=2 (30s) |
| Train/holdout session split | **PASS** | 70/30 sessions; overlap gate PASS |
| Benign-only train + scaler | **PASS** | Leakage-aware fit |
| Binary + multiclass labels | **PASS** | MC sanity PASS |
| `ts_val` / `ts_test` in MC npz | **PASS** | Present in `windows_vnext_mc.npz` |
| Binary `windows_vnext.npz` timestamps | **WARNING** | Drift loads binary file; may lack `ts_*` unless binary preprocess re-run |
| Val/test split | **WARNING** | Window-level 50/50; sessions can appear in both |

**Shapes (run-final):** train benign `(16795, 10, 163)`; val/test `(5153, 10, 163)`; holdout attack rate ~38.2%.

---

## 5. Model issues (detail)

| Item | Status |
|------|--------|
| Architecture (SequenceBottleneckAE) | Implemented |
| Contractive noise, denoising, AUC early-stop | Executed |
| Multi-seed {42, 7, 1337} | Executed → 0.7529 ± 0.023 |
| Optuna HPO (12 trials) | Executed → 0.8138 |
| Per-attack §19 | Executed (named labels) |
| Ensemble | Computed then discarded (worse than AUC-best) |
| §18b ablations | **Not executed** |
| Deep baselines in model notebook | **Missing** |

---

## 6. Drift issues (detail)

| Item | Status |
|------|--------|
| `ALERT_INVERT=False` on saved scores | Correct |
| Offline thr from `metrics_vnext.json` | Correct (`−0.177461`) |
| Full-test integrity | **PASS** |
| Stream integrity | **PASS** (FPR 0.2345) |
| Ordering | `container_lexsort` (not timestamp) |
| Adaptive last-half | FPR↓ but recall/F1 collapse → `claim_ok=False` |
| Matched FPR comparison | Present; does not support adaptive win |
| Fine-tune (this freeze) | +0.0319 AUC (conflicts with older −0.06 narrative) |
| Isolation Forest baseline | Present (~0.674) |
| Invalid v1 FPR reduction claim | Correctly rejected |

---

## 7. Improvement plan

### Phase A — Before defense (mandatory)

| # | Action | Closes | Owner effort |
|---|--------|--------|--------------|
| A1 | Relock docs to `run-final` numbers **or** re-run to restore 0.7402 and freeze that run | ISS-01 | Docs + 1 re-run |
| A2 | Rewrite Abstract/Results: multi-seed in Abstract; default as Methods; HPO as best-case only | ISS-01, ISS-05 | Writing |
| A3 | Explicitly state stream ordering = `container_lexsort` until timestamps fixed | ISS-06 | Writing + optional re-run |
| A4 | Forbid adaptive “improvement” wording unless `claim_ok=True` | ISS-02, ISS-03 | Writing |
| A5 | Re-run `mdc_preprocess_vNext.ipynb` so `windows_vnext.npz` has `ts_val`/`ts_test`; re-run drift; confirm `ordering_mode=timestamp` | ISS-06 | Colab |
| A6 | Set `RUN_ABLATIONS=True` on GPU; fill ablation Table A | ISS-08 | Colab GPU |
| A7 | Add ≥1 deep baseline (LSTM AE or dense AE) on **same** windows; keep IF | ISS-09 | Code + run |
| A8 | Unify scoring protocol (same `feat_std` / invert / ensemble policy) across default, multi-seed, HPO, drift | ISS-04 | Code + re-run |
| A9 | Freeze fine-tune narrative to **this** freeze’s sign, or re-run and lock one result | ISS-10 | Docs / re-run |
| A10 | Session-disjoint val/test **or** nested threshold tuning; document choice | ISS-07 | Code + re-run |

### Phase B — Nice-to-have (before submission if time)

| # | Action | Closes |
|---|--------|--------|
| B1 | Report TPR @ FPR = 5/10/25% with interpretation | ISS-13 |
| B2 | Latency table with hardware note (CPU model) | ISS-14 |
| B3 | State ensemble was evaluated and rejected | ISS-11 |
| B4 | Per-attack: emphasize n≥50 only; mark sparse classes | ISS-12 |
| B5 | Detection delay / time-to-detect on stream | ISS-20 |
| B6 | Prefer zip download for Drive artifacts (document in README) | ISS-15 |
| B7 | Mark `mdc_dataloader.ipynb` as LEGACY in notebook header | ISS-16 |

### Phase C — Future work (thesis chapter, not required for pass)

| # | Action | Closes |
|---|--------|--------|
| C1 | Regularized / replay-buffered online adaptation | Adaptation gap |
| C2 | Second dataset / external validation | ISS-18 |
| C3 | Explainability (attribution + PSI drivers) | ISS-19 |
| C4 | Live or near-real-time deployment study | ISS-17 |
| C5 | Paired statistical tests vs Exp A / IF | Methodology depth |

---

## 8. Claim policy (what you may / may not say)

| Claim | Allowed? | Condition |
|-------|----------|-----------|
| Transformer AE detects MDC anomalies | **Yes** | Cite multi-seed ± HPO; default per freeze |
| Preprocess is leakage-free (train/holdout sessions) | **Yes** | Session gate PASS |
| Holdout shows distribution shift (PSI) | **Yes** | Model §14 / drift baseline |
| Simulated streaming works | **Yes** | With ordering mode stated |
| CPU inference &lt; 100 ms/window | **Yes** | Cite measured ms + hardware |
| IF weaker than AE | **Yes** | 0.674 vs default/multi-seed |
| Adaptive thresholding improves monitoring | **No** (this freeze) | Only if `claim_ok=True` later |
| Fine-tune improves drift detection | **No** (unstable) | Lock one freeze first |
| Live deployment validated | **No** | — |
| v1 FPR reduction ~92% | **Never** | Invalid |

---

## 9. Recommended thesis wording updates

### Abstract (template aligned to `run-final`)

> We present a leakage-aware Transformer autoencoder for container network anomaly detection on the MDC dataset, achieving **0.7529 ± 0.023** ROC-AUC across three seeds (best HPO **0.8138**; default freeze **0.7042**). We evaluate drift monitoring (PSI/KS) and drift-aware mechanisms under **simulated streaming replay** (`container_lexsort` unless timestamps are available). Adaptive thresholding did not improve monitoring at matched operating points without recall collapse; online benign fine-tuning results are reported but not claimed as successful adaptation.

### Methods — threshold

> Primary operating point is the validation-tuned F1-optimal threshold from `metrics_vnext.json` for the frozen run (this freeze: **≈ −0.1775**), applied with `ALERT_INVERT=False` on already-oriented scores.

### Discussion — drift

> Results support **detection of shift**, not **successful online adaptation**, under the current integrity-gated evaluation.

---

## 10. Checklist before viva

- [ ] ISS-01: Docs and Abstract match `run-final` (or a newly frozen superior run)
- [ ] ISS-02/03: No adaptive/v1 FPR success claims
- [ ] ISS-06: Ordering mode stated; preferably timestamp after re-export
- [ ] ISS-08: Ablation table filled or explicitly scoped out with justification
- [ ] ISS-09: At least IF + one deep/classical baseline under same data
- [ ] ISS-04: Scoring protocol consistency documented
- [ ] ISS-07: Val/test leakage risk acknowledged or fixed
- [ ] ISS-10: Fine-tune narrative matches frozen evidence
- [ ] Freeze pack: `metrics_vnext.json`, `matched_policy_comparison.json`, `freeze_manifest_v2.json`, thesis figures

---

## 11. Priority backlog (ordered)

1. Relock metrics / rewrite claims (ISS-01, ISS-02, ISS-05)  
2. Timestamp re-export + drift re-run (ISS-06)  
3. Run ablations (ISS-08)  
4. Add deep baseline (ISS-09)  
5. Unify scoring protocol (ISS-04)  
6. Fix or disclose val/test session split (ISS-07)  
7. Lock fine-tune story (ISS-10)  
8. Nice-to-haves (Phase B)  
9. Future work (Phase C)

---

## 12. Document history

| Date | Change |
|------|--------|
| 2026-07-13 | Initial register from M4 research-evaluation-analyst audit of `run-final` notebooks |
