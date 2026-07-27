# Module 4: Security Anomaly Detector — Drift-Aware Approach
## Independent Research Validation Report

**Reviewer stance:** Independent, evidence-first audit (senior research-analyst lens, 10y equivalent)
**Evidence base:** 5 executed notebooks (`module4/notebook/final/kaggle-output/`) + research-record docs 01–05 + `claim_evidence_matrix.md` + `ISSUES_AND_IMPROVEMENT_PLAN.md` + thesis notes
**Canonical metrics freeze referenced:** `module4/notebook/final/output-metrics/model_vnext_runs/metrics_vnext.json` (2026-07-25)

---

## Verdict

**Overall score: 68 / 100 — Defensible for a BSc final-year submission, with mandatory rewrites before viva.**

The underlying detector and evaluation harness are built to a standard well above typical FYP rigor — but the thesis wording must be brought into line with what the code actually proves. Two of the three headline "novelties" are implemented as instrumented, honestly-reported **negative or inconclusive** findings rather than demonstrated improvements, and the proposal's multi-modal telemetry (CPU/memory/syscalls/file access) never made it into the frozen pipeline, which is network-flow-only.

---

## Novelty Scorecard

| # | Novelty (as proposed) | Score | Status | One-line verdict |
|---|---|---:|---|---|
| 1 | Drift-aware detection (sliding window + incremental learning + error-based drift) | **55/100** | Partial | Mechanism fully built & rigorously tested; the claimed *benefit* is not shown — the project's own freeze gate marks it `claim_ok=false`. |
| 2 | Zero-day handling via high reconstruction error | **50/100** | Partial | True by construction (benign-only training), but no dedicated held-out-attack-family experiment exists to prove genuine generalization. |
| 3 | Multi-metric correlation via attention (resource + security signals) | **30/100** | Largely unsupported | Attention mechanism is real, but it correlates only network-flow sub-features. CPU, memory, syscall and file-access signals are absent from the frozen dataset. |

Completion score reflects two things: (a) is the mechanism actually implemented and executed, and (b) does the evidence support the *specific benefit claimed* in the proposal — not merely that code exists.

---

## 1. Foundation check — is the core detector itself sound?

Before scoring the three novelties, the base claim needs to hold: an unsupervised Transformer autoencoder, trained only on benign windows, that separates attack from benign traffic without leakage.

### Leakage-free preprocessing — ✅ Verified
Container sessions are split 70/30 *before* any scaler is fit; both StandardScalers (flow-level, then 15s-bucket-level) are fit on train-benign only; the manifest records `leakage_free: true`.

> **Evidence:** `module4/notebook/final/output-metrics/windows_vnext_processed/manifest_vnext.json` — shapes `X_train (16795,10,163)`, `X_val/X_test (5153,10,163)`, attack rate 38.2%. Pipeline stages documented in `docs/research-record/02_data_pipeline_and_methodology.md` §2.1.

### Architecture — ✅ Verified, real attention mechanism
`SequenceBottleneckAE`: linear projection → sinusoidal positional encoding → `nn.TransformerEncoder` (multi-head self-attention, 4 heads) → learned bottleneck (dim 24 default / 16 after HPO) → `nn.TransformerDecoder` with learned positional queries → linear output. This is a genuine attention-based encoder-decoder, not a rebranded MLP.

> **Evidence:** `kaggle_drift_aware_2607.ipynb`, code cell defining `class SequenceBottleneckAE(nn.Module)` (loaded from checkpoint, confirms 270,571 params, `d=64 bn=24 ff=256 enc=2 dec=2`).

### Headline detection metric — ⚠️ Real, but which number is "final" is currently ambiguous
Three different "final" ROC-AUC lineages exist across the project's history, and older docs still lock the oldest one.

| Era | Default ROC-AUC | Multi-seed | HPO best | Still cited in |
|---|---:|---:|---:|---|
| REFERENCE lat (2026-07-08) | 0.7402 | 0.7529 ± 0.023 | 0.8138 | `claim_evidence_matrix.md`, `thesis_drift_aware_notes.md`, `thesis_results_discussion.md` |
| Jul-24 integrity archive | 0.7042 | 0.7529 ± 0.023* | 0.8138* | `ISSUES_AND_IMPROVEMENT_PLAN.md` |
| **Final Kaggle freeze (canonical)** | **0.6889** | **0.7261 ± 0.030** | **0.8446** | `output-metrics/model_vnext_runs/metrics_vnext.json` |

\* Flagged in the project's own issues register as possibly carried over rather than recomputed — treat with care.

The frozen `freeze_manifest_v2.json` even embeds the old 0.7402/0.7529/0.8138 numbers as `headline_metrics_locked`, which now **contradicts** the very metrics file sitting next to it. This is the single most important item to fix before submission — see §6.

---

## 2. Novelty 1 — "True drift-aware detection combining sliding window, incremental learning, and error-based drift handling"

### Sliding window (500–1000 samples) — ✅ Verified
Implemented as a 500-sample benign-score calibration buffer for the adaptive threshold — explicitly sized to match the proposal's 500–1000 range.

> **Evidence:** `kaggle_drift_aware_2607.ipynb`, config cell: `BENIGN_BUFFER_SIZE = 500  # aligned to proposal's stated 500-1000 sample window`

**Caveat:** this buffer calibrates the *threshold*, not the model's input context — the AE itself still consumes fixed T=10 buckets (150s) per window. A reader who takes the proposal's "sliding window for current behaviour focus" to mean the model's receptive field will find a smaller number (10, not 500–1000). Worth one clarifying sentence in the thesis so the two window concepts aren't conflated.

### Error monitoring with statistical tests (KS-test) — ⚠️ Partial
Implemented, but not a textbook KS-test. The code computes PSI (population stability index, primary drift signal) and a "KS" statistic per feature — the latter is the max absolute difference between the current batch's empirical quantiles and the frozen baseline quantiles, not `scipy.stats.ks_2samp` with a p-value. It approximates the KS statistic's spirit without the significance-testing apparatus the name implies.

> **Evidence:** `kaggle_model_2607.ipynb` §14 "Drift monitor (PSI + KS on holdout vs benign train baseline)" — functions `psi_per_feature`, `ks_per_feature`; output saved to `drift_report_vnext.npz` (keys `psi`, `ks`, length 163). Direct PSI-over-time cross-check independently recomputed in `kaggle_analysis_final.ipynb` §4.

Rename this "a KS-inspired quantile-shift statistic" in the write-up — a supervisor who runs `scipy.stats.ks_2samp` against the code and doesn't find it will read the mismatch as overclaiming, even though the underlying signal (PSI, confirmed present with dozens of features >0.25 across most 2023–2024 time buckets) is genuinely detecting shift.

### Adaptive thresholding — ❌ Unsupported (as an improvement)
Implemented with real engineering care — alert-space unification, an integrity gate that specifically catches the exact double-invert bug that invalidated an earlier project run — but the final freeze's own gate says the adaptive mechanism does *not* work on this data.

| Operating point (last-half stream) | FPR | Recall | F1 |
|---|---:|---:|---:|
| Fixed threshold (offline f1_optimal) | 0.4651 | 0.7329 | 0.7783 |
| Adaptive threshold (median + 2.5σ, buffer 500) | 0.0000 | 0.0000 | 0.0000 |

The adaptive rule pushes its threshold so high it stops alerting entirely in the drift half of the stream — a real, honestly-reported failure mode, not a bug being hidden. `matched_policy_comparison.json → claim_adaptive_improves_without_recall_loss: false`, interpretation string: `"tradeoff_or_negative — do not claim FPR reduction alone"`.

### Incremental / online learning — ❌ Unsupported (as an improvement)
Implemented (50-step benign-only MSE fine-tune on the last-30% "drift" slice) but the final freeze's result is undefined, and the project's own historical run showed it makes detection *worse*.

| Run | AUC before | AUC after | Δ | Note |
|---|---:|---:|---:|---|
| Legacy v1 (historical) | 0.8508 | 0.7902 | −0.0605 | Catastrophic forgetting |
| Final Kaggle freeze | NaN | NaN | NaN | `stable_slice_fallback` — chronological drift slice was 100% attack, so AUC is undefined |

The freeze manifest's own `do_not_cite` list includes `"fine-tune improves detection"`. This is a legitimate, well-diagnosed research finding (unsupervised fine-tuning on recent benign traffic erodes the attack-separation boundary — classic catastrophic forgetting) and should be written up as exactly that in the Discussion chapter, not softened into a success.

### Novelty 1 — bottom line
**Score: 55/100.** Every mechanism the proposal names is present in code, executed, and evaluated with unusual rigor (there is a dedicated integrity gate whose entire purpose is to catch a specific historical scoring bug — that is genuinely good engineering practice). But "drift-aware" as proposed implies the system *copes better* under drift than a static baseline, and on this dataset that specific claim is not supported — it is actively contradicted by the project's own claim gates. Frame this novelty in the thesis as "we built and rigorously evaluated a drift-aware monitoring layer; we found threshold adaptation and weight fine-tuning both fail to improve on the static baseline, which is itself a useful negative result," not as "our drift-aware mechanism improves detection."

---

## 3. Novelty 2 — "Zero-day attack handling through high reconstruction error on unseen patterns"

### True by construction — ✅ Verified
`X_train` (16,795 windows) is 100% benign — no attack window, of any type, is ever shown to the model during training. Any attack the model flags is, by definition, an "unseen pattern" from the model's perspective. This is the standard argument for benign-only autoencoder anomaly detection, and it is correctly implemented here.

> **Evidence:** `manifest_vnext.json`: NPZ contains no `y_train` key at all (only val/test carry labels). `docs/research-record/02_data_pipeline_and_methodology.md` §1 confirms.

### No dedicated generalization experiment — ⚠️ Partial
A stronger "zero-day" demonstration would hold out an entire attack *family* from both training and threshold-tuning, then show the detector still fires on it at test time. That experiment isn't in this freeze. The per-attack-type breakdown that would let a reader judge coverage breadth (`eval_multiclass.json`) is also absent from the current freeze.

> **Evidence:** `kaggle_analysis_final.ipynb` §8: `"Need eval_multiclass.json + the timeline dataframe... skipping cross-reference."` Historical per-attack numbers exist only in older, superseded runs, and several classes there have `n ≤ 19` (statistically unreliable) — e.g. Node-RED RCE and Node-RED Escape have exactly 1 window each in the reference split.

### Operating-point honesty check — ⚠️ Partial
At the primary (F1-optimal) threshold in the final freeze, the detector runs at FPR ≈ 0.29 — meaning roughly 3 in 10 benign windows would raise a false alarm. The HPO-tuned configuration cuts this to FPR ≈ 0.055, but that is reported as a best-case ceiling, not the operating default.

> **Evidence:** `metrics_vnext.json → test_default.f1_optimal` (FPR 0.2911) vs `test_hpo` (FPR 0.0546).

### Novelty 2 — bottom line
**Score: 50/100.** The mechanism is correctly built and the "unseen pattern" argument holds by design — this part is not in question. What's missing is the evidence a marker would actually want to see: a named per-attack-type recall table, and ideally one held-out-family generalization test, to turn "structurally plausible" into "empirically demonstrated." At a ~0.29 FPR operating point, "handles zero-day attacks" should be phrased as "surfaces unseen attack patterns at a moderate, quantified false-alarm rate" rather than an unqualified capability claim.

---

## 4. Novelty 3 — "Multi-metric correlation monitoring via attention and joint error analysis... across resource and security signals"

### The attention half is real — ✅ Verified
The Transformer encoder/decoder genuinely attends jointly across all 163 features and all 10 timesteps; the anomaly score itself is a joint error analysis — a weighted blend of the mean reconstruction error (0.3) and the time-max of the mean-over-feature error (0.7), i.e. it already combines a "typical behaviour" signal with a "worst single moment" signal across the whole feature set.

> **Evidence:** `kaggle_drift_aware_2607.ipynb`, function `_combine(err, mean_w=0.3, max_w=0.7)`; same formula documented in `docs/research-record/02` §4.6.

### The "resource and security signals" half is not implemented — ❌ Unsupported
The proposal's objective explicitly names CPU, memory, network flows, system calls, and file-access logs as the inputs to correlate. The frozen pipeline's 163 features are *exclusively* CICFlowMeter-style network-flow statistics (durations, packet/byte counts, flags, inter-arrival timings, etc.). There are no CPU, memory, syscall, or file-access columns anywhere in `windows_vnext.npz`.

> **Evidence:** `docs/research-record/01_project_overview_and_architecture.md` §1, explicit self-audit line: *"CPU/memory/syscall/file-access columns... Not found in final NPZ / feat_names."* Confirmed independently by inspecting `drift_baseline_vnext.npz → feat_names` (163 flow-derived names, no resource/syscall terms).

This is the widest gap between the proposal and the delivered system, and it is the project's own audit that surfaces it first — that transparency is a strength of the documentation, but the gap itself is real and material: "coordinated attacks across resource and security signals" specifically implies cross-domain correlation (e.g. a CPU spike *plus* an unusual syscall pattern *plus* a network beacon, correlated together). What's actually demonstrated is correlation *within* one domain (many network-flow statistics attended to jointly), which is a real and useful thing, but a narrower claim.

### Novelty 3 — bottom line
**Score: 30/100.** Don't drop this novelty — reframe it. "Attention-based joint reconstruction-error analysis across 163 correlated network-flow features" is true, implemented, and defensible. "Multi-metric correlation across resource and security signals to catch coordinated attacks" is not implemented and should either be (a) rescoped in the abstract/objective to explicitly state network-flow telemetry only, with cross-domain fusion named as future work, or (b) if time permits before submission, partially addressed by adding even one non-flow signal (e.g. a lightweight CPU/memory time series aligned to the same containers) as a second input stream — a substantial scope addition this late, so (a) is the realistic path.

---

## 5. What's genuinely strong — don't undersell this in the viva

Novelty aside, the project's *research hygiene* is unusually good for undergraduate work, and it's worth stating this to the examiners explicitly, because it's a real point in the project's favour that a scorecard on "novelty" alone won't capture.

- **A dedicated integrity gate that catches a real historical bug.** `validate_threshold_integrity()` exists specifically because an earlier run (v1) recomputed the alert threshold with the wrong invert convention and produced a fabricated "92% FPR reduction." The final pipeline now refuses to run adaptive comparisons unless this gate passes, and the invalid v1 number is tracked in a permanent `do_not_cite` list. Catching and permanently quarantining your own past mistake, in code, is exactly the kind of self-correction examiners want to see.
- **A living claim-evidence matrix** (`docs/claim_evidence_matrix.md`) that tags every claim SUPPORTED / CONDITIONAL / NOT SUPPORTED / INVALID with its exact evidence source — this is close to how a research lab tracks pre-registration vs. results, well beyond typical FYP documentation.
- **Fair, protocol-matched baselines.** Isolation Forest and a Dense AE are evaluated on *identical* windows with the *identical* threshold rule and auto-invert-flip check as the main model — this is correct experimental control, not a strawman baseline.
- **Honest negative results.** The fine-tune and adaptive-threshold sections could easily have been quietly omitted; instead they're reported, gated, and explicitly marked non-citable. That's more scientifically valuable than a suspiciously clean success story.

---

## 6. Priority action list before submission / viva

| Priority | Action | Why |
|---|---|---|
| **Critical** | Reconcile the three metric eras into one number the thesis actually cites. | Pick the final Kaggle freeze (0.6889 default / 0.7261±0.030 multi-seed / 0.8446 HPO) as canonical, then update `claim_evidence_matrix.md`, `thesis_drift_aware_notes.md`, `thesis_results_discussion.md`, and `freeze_manifest_v2.json`'s embedded `headline_metrics_locked` field to match. An examiner cross-checking your Abstract against your own frozen JSON is the single easiest inconsistency to get caught on. |
| **Critical** | Rewrite the three novelty claims to match what's proven, not what's implemented. | Drift-adaptation and zero-day framing should read as "implemented and rigorously evaluated; adaptation found not to help (catastrophic forgetting / threshold collapse), a useful negative result" rather than as delivered improvements. |
| **High** | Explicitly rescope the objective/abstract to network-flow telemetry only. | State plainly that CPU/memory/syscall/file-access fusion is future work, not delivered — this pre-empts the most obvious viva question ("where's the multi-modal data your proposal promised?"). |
| **High** | Produce the per-attack-type recall table for the current freeze. | Re-run model notebook §19 / export `eval_multiclass.json` so `kaggle_analysis_final.ipynb` §8's cross-reference actually populates — right now it prints "skipping" because the file is missing from this freeze. |
| **Medium** | Rename the "KS-test" to what it is. | Call it a KS-inspired quantile-shift statistic in the Methods chapter, distinct from PSI, so a supervisor checking for `scipy.stats.ks_2samp` doesn't flag a mismatch. |
| **Medium** | Pin dependency versions. | No `requirements.txt`/lockfile exists under `module4/`; add one from the Kaggle environment for reproducibility credit. |

---

*Prepared as an independent evidence-first audit against `module4/notebook/final/kaggle-output/*.ipynb` (executed cell outputs) and `module4/docs/` (research-record 01–05, claim-evidence matrix, issues register, thesis notes). Where the project's own documentation already flags a gap, that is credited rather than re-discovered — several findings above quote the project's own `research-record` audit directly.*
