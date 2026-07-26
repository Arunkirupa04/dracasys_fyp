# Thesis Chapter Draft — Results, Discussion, Limitations

**Module 4 — Security Anomaly Detector (MDC / Transformer AE vNext)**  
**Status:** Draft aligned to Tasks 7–10 (2026-07-11)  
**Headline lock:** [`claim_evidence_matrix.md`](claim_evidence_matrix.md)  
**Evidence freeze:** `notebook/run/mdc_drift_aware_output_v2.ipynb` (after Colab re-run)

---

## 1. Results

### 1.1 Core detection performance

The default SequenceBottleneckAE (vNext) achieves **ROC-AUC 0.7402** on the held-out test set at the validation-tuned F1-optimal threshold (F1 ≈ 0.741, MCC ≈ 0.588), improving on the prior Exp A baseline (0.7163). Across three seeds {42, 7, 1337}, mean ROC-AUC is **0.7529 ± 0.023**, supporting robustness rather than a single lucky run. Hyperparameter optimization yields a best-case test ROC-AUC of **0.8138** (MCC ≈ 0.626, F1 ≈ 0.774); this is reported as an upper bound, not the sole Abstract claim.

| Config | ROC-AUC | F1 | MCC | Role in thesis |
|--------|---------|----|-----|----------------|
| Exp A | 0.7163 | — | — | Prior baseline |
| vNext default | **0.7402** | 0.741 | 0.588 | Methods reference |
| Multi-seed mean | **0.7529 ± 0.023** | — | — | Abstract robustness |
| vNext HPO | **0.8138** | 0.774 | 0.626 | Best-case |
| Isolation Forest (mean+max) | ~0.674 | — | — | Classical baseline |

Bootstrap 95% CI on the default test ROC (reference drift run) was approximately **[0.725, 0.757]**, consistent with stable ranking above Exp A.

### 1.2 Per-attack-type analysis (Task 10)

Using official MDC labels (Sever & Dogan, 2023), the reference test split contains sparse multiclass support:

| Label | Name | Windows (ref.) | Notes |
|------:|------|---------------:|-------|
| 0 | BENIGN | 3184 | FPR operating point |
| 1 | CVE-2020-13379 | 1679 | Dominant attack class |
| 2 | Node-RED Recon | 266 | Secondary class |
| 3 | Node-RED RCE | 1 | Not statistically reliable |
| 4 | Node-RED Escape | 1 | Not statistically reliable |
| 8 | CVE-2021-25741 | 19 | Low support |
| 11 | DSB Nuclei Scan | 3 | Low support |

**Interpretation rule:** Emphasize classes with *n* ≥ 50. Low-*n* rows are reported for completeness only. After re-running §19 with the updated label map, paste recall/AUC from `eval_multiclass.json`.

### 1.3 Drift detection

PSI/KS monitoring against the benign-train drift baseline shows substantial feature-level shift on holdout traffic (reference: dozens of features with PSI > 0.25). This supports the claim that **distribution shift is detectable**, which is expected under a random session split that mixes attack traffic into holdout.

### 1.4 Drift-aware mechanisms (after Tasks 1–3 re-run)

Report only from `mdc_drift_aware_output_v2.ipynb` / `matched_policy_comparison.json`:

- **Fixed threshold:** offline thr ≈ −0.1434, `ALERT_INVERT=False`, integrity gate PASS.
- **Adaptive threshold:** compare F1/recall/MCC at matched FPR; use `claim_ok` before claiming improvement.
- **Fine-tune:** reference run showed drift-slice AUC **decrease** (~−0.06) after benign-only MSE fine-tuning → catastrophic forgetting.

### 1.5 Ablations (Task 7–8)

**Table A (offline)** — same protocol only:

| Config | ROC-AUC | Δ vs default | Source |
|--------|---------|--------------|--------|
| vNext default | 0.7402 | 0 | metrics |
| HPO | 0.8138 | +0.0736 | metrics |
| Multi-seed mean | 0.7529 | +0.0127 | metrics |
| IF mean+max | ~0.674 | −0.066 | drift §8 |
| no_contractive | *fill after §18b* | *Δ* | `metrics_ablation_vnext.json` |
| no_early_stop | *fill after §18b* | *Δ* | `metrics_ablation_vnext.json` |

**Table B (stream)** — do not mix with Table A metrics in one claim sentence. Export: `ablation_tables.md`.

### 1.6 Deployment feasibility

CPU inference latency ≈ **3.36 ms/window** on the reference hardware profile, well below interactive/monitoring budgets.

---

## 2. Discussion

### 2.1 What the detector contributes

The main contribution is a **leakage-aware, reproducible Transformer AE pipeline** for container-network misuse detection, with stronger offline metrics than Exp A and a fair classical baseline (Isolation Forest). Novelty is primarily in **integration and evaluation discipline**, not a new neural operator.

### 2.2 Drift detection vs drift adaptation

Results support a clear distinction:

1. **Detection of shift (PSI/KS)** — supported.
2. **Adaptation via weight updates** — not supported under naive benign fine-tuning.
3. **Adaptation via threshold recalibration** — plausible operational response; must be argued only with matched operating-point evidence.

This negative fine-tune result is scientifically useful: it shows that “drift-aware” must not be equated with “online learning succeeds.”

### 2.3 Ablation expectations (fill after GPU run)

- If `no_contractive` drops ROC/F1 → contractive term helps representation stability.
- If `no_early_stop` matches or worsens test metrics → early-stop is justified as regularization / restore policy.
- If deltas are within multi-seed noise (~0.02) → claim “no strong effect” rather than over-interpreting.

### 2.4 Per-attack asymmetry

High recall on CVE-2020-13379 with weak recall on rare classes is expected under class imbalance and window aggregation. Thesis should discuss **coverage vs rarity**, not claim uniform detection across all CVEs.

---

## 3. Limitations

1. **Single dataset (MDC)** — no cross-dataset generalization study.
2. **Simulated streaming** — ordered holdout replay, not a live cluster deployment.
3. **Temporal validity** — requires `ts_test` re-export (Task 6); otherwise container-lexsort is a proxy.
4. **Threshold tuned on validation** — risk of optimistic operating-point selection; mitigated by reporting ROC/PR and multi-seed AUC.
5. **Sparse multiclass labels** — several attack IDs have *n* ≤ 20 in test.
6. **Fine-tune protocol is naive** — no EWC/replay/adapters; negative result may not generalize to regularized adaptation.
7. **Ablation GPU runs** — thesis numbers require `RUN_ABLATIONS=True` (Task 7); smoke mode is not usable.

---

## 4. Threats to Validity

| Threat | Risk | Mitigation |
|--------|------|------------|
| **Internal — data leakage** | Scaler/fit seeing holdout | Session split; benign-only fit; preprocess gates |
| **Internal — threshold mismatch** | Invalid adaptive comparison | Load thr from metrics; `ALERT_INVERT=False`; integrity gate |
| **Internal — metric mixing** | Comparing ROC to stream FPR | Split Table A / Table B |
| **Construct — “real-time”** | Overclaiming deployment | “Simulated streaming replay” wording |
| **External — dataset shift** | Results MDC-specific | Explicit single-dataset limitation |
| **Statistical — rare classes** | Unstable recall | Report *n*; emphasize large classes |
| **Conclusion — adaptation** | Claiming fine-tune success | Report negative Δ; claim matrix C8 = NOT SUPPORTED |

---

## 5. Future work

- Regularized online adaptation (EWC, replay buffer, LoRA/adapters)
- Additional classical/deep baselines (OCSVM, dense AE)
- Cross-dataset or cross-cluster evaluation
- True temporal deployment study with production timestamps
- Cost-sensitive thresholds for SOC false-positive budgets

---

## 6. Paste-in checklist before submission

- [ ] Replace ablation placeholders with §18b numbers (`thesis_usable=true`)
- [ ] Paste Table A/B from `ablation_tables.md` after v2 freeze
- [ ] Paste per-attack recalls from updated `eval_multiclass.json` (named CVEs)
- [ ] Confirm Abstract uses **0.7529 ± 0.023** (and mentions HPO only as best-case)
- [ ] Confirm no citation of v1 `fpr_reduction ≈ 0.92`
