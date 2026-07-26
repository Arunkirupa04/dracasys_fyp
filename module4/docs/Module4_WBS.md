# Module 4 — Security Anomaly Detector
# Work Breakdown Structure (WBS)
# Proposal alignment: Current State → Remaining Implementation → Thesis

**Last updated:** 2026-07-11  
**Reference run:** `notebook/run/mdc_model_vNext_lat_output.ipynb`  
**Drift-aware v2:** `notebook/mdc_drift_aware.ipynb` + `notebook/mdc_eval_utils.py`  
**P0 Tasks 1–3:** threshold integrity + matched-FPR coded; **Colab re-run required**  
**P0/P1 Tasks 4–6:** freeze infra + claim matrix locked + timestamps coded in preprocess `_mc`  
**Drive root (all notebooks):** `My Drive / vNEXT_test / {processed|runs|drift_aware}` — see `docs/DRIVE_vNEXT_test.md`  
**Claim lock:** `docs/claim_evidence_matrix.md`  
**Run registry:** `notebook/run/README_RUNS.md`

---

## Current Status Summary (as of latest run)

| Item | State | Evidence |
|------|-------|----------|
| `mdc_preprocess_vNext.ipynb` | **DONE** | `windows_vnext.npz`, scale PASS, manifest |
| `mdc_preprocess_vNext_mc.ipynb` | **DONE** | `windows_vnext_mc.npz`, alignment PASS |
| `mdc_model_vNext.ipynb` | **DONE** | Full pipeline + §19 multiclass in one notebook |
| Default test ROC-AUC | **0.7402** | Beats Exp A (0.7163) |
| Multi-seed ROC-AUC | **0.7529 ± 0.023** | Seeds 42, 7, 1337 |
| HPO test ROC-AUC | **0.8138** | MCC 0.626, F1 0.774 |
| Drift baseline export | **DONE** | `drift_baseline_vnext.npz` |
| Drift monitor (PSI/KS) | **DONE** | 99/163 features PSI > 0.25 |
| Per-attack-type eval (§19) | **DONE** | `eval_multiclass.json` + plots |
| **Adaptive thresholding** | **FIXED (Tasks 1–2)** | Offline thr + `ALERT_INVERT=False` + matched-FPR |
| **Incremental fine-tune** | **IMPLEMENTED** | §6 — report negative Δ honestly |
| **Streaming simulation** | **IMPLEMENTED (v2)** | §7 — timestamp or container replay |
| **Baseline comparison** | **IMPLEMENTED (v2)** | IF benign-only mean+max §8 |
| **Ablation study** | **PARTIAL** | Split offline/stream tables; §18b optional |
| **Thesis figures package** | **IMPLEMENTED** | §11 + matched-FPR fig07b |
| **Window timestamps** | **IMPLEMENTED** | `ts_val`/`ts_test` in preprocess §11 |
| **Threshold integrity (P0)** | **CODED — needs Colab re-run** | Task 3: freeze `mdc_drift_aware_output_v2.ipynb` |
| **Freeze / legacy registry (Task 4)** | **INFRA READY** | `freeze_manifest_v2.json` + `README_RUNS.md` + LEGACY banner |
| **Claim–evidence lock (Task 5)** | **DONE** | `docs/claim_evidence_matrix.md` |
| **Window timestamps (Task 6)** | **CODED in vNext + _mc** | Re-export npz → `ordering_mode=timestamp` |
| **Retrain ablations (Task 7)** | **HARNESS READY** | Set `RUN_ABLATIONS=True` on Colab GPU |
| **Ablation tables A/B (Task 8)** | **DONE** | CSV + `ablation_tables.md` in drift-aware |
| **Thesis Results/Discussion (Task 9)** | **DONE (draft)** | `docs/thesis_results_discussion.md` |
| **Official attack names (Task 10)** | **DONE (code)** | `mdc_label_map.py` + §19; re-run for plots |

---

## Proposal Claims vs Implementation

| Proposal claim | What you have | Gap |
|----------------|---------------|-----|
| Transformer AE anomaly detection | SequenceBottleneckAE vNext | None |
| Robust preprocessing pipeline | vNext preprocess + mc labels | None |
| Strong offline evaluation | ROC/MCC/F1/multi-seed/HPO | Fill results tables in thesis |
| Drift **detection** (PSI/KS) | §14 monitor + baseline | None |
| Drift-**aware** adaptive threshold | Not coded | **Implement §3.2** |
| Incremental / online adaptation | Not coded | **Implement §3.3** |
| Real-time streaming evaluation | Not coded | **Implement §5.0** |
| Per-threat-type analysis | §19 in model notebook | Map real attack names |
| Baseline comparison | Not run | **Implement §4.3** |

---

## 0.0 COMPLETED — Do Not Rebuild (Archive & Cite)

### 0.1 Preprocessing (`mdc_preprocess_vNext.ipynb`) — DONE
- [x] Random session split, benign-only fitting
- [x] StandardScaler flow + bucket, T=10, mean_max_std
- [x] Short-gap fill, manifest, drift baseline
- [x] Artifacts: `windows_vnext.npz`, `drift_baseline_vnext.npz`, `manifest_vnext.json`

### 0.2 Multiclass preprocess (`mdc_preprocess_vNext_mc.ipynb`) — DONE
- [x] `label_multiclass` in bucket + window
- [x] `y_val_multiclass`, `y_test_multiclass` in `windows_vnext_mc.npz`

### 0.3 Model training (`mdc_model_vNext.ipynb`) — DONE
- [x] Train, score-flip, AUC early stop, contractive reg
- [x] Multi-seed, HPO, default + HPO eval
- [x] Drift monitor §14, save §17, persist §17b
- [x] Per-attack eval §19 (merged — no separate eval notebook)
- [x] Latest metrics recorded in `mdc_model_vNext_lat_output.ipynb`

**Action:** Export key numbers to §7.1 template below. No retrain unless doing ablations.

---

## 1.0 PRIORITY 1 — Thesis Results Packaging (No New Code)

**Goal:** Turn existing run into thesis-ready evidence.  
**Effort:** 1–2 days.  
**Blocked by:** Nothing.

### 1.1 Lock headline numbers
- [x] 1.1.1 Abstract: HPO ROC **0.8138**, robust **0.7529 ± 0.023**
- [x] 1.1.2 Methods: default **0.7402** vs Exp A **0.7163**
- [x] 1.1.3 Document score inversion (`SCORE_INVERT=True`, negative thresholds)
- [x] 1.1.4 Claim–evidence matrix locked (`docs/claim_evidence_matrix.md`)

### 1.2 Fill results tables (from `mdc_model_vNext_lat_output.ipynb`)
- [ ] 1.2.1 Main table — all threshold strategies (§11 output)
- [ ] 1.2.2 Multi-seed table (§13 output)
- [ ] 1.2.3 HPO vs default comparison (§16 output)
- [ ] 1.2.4 History/ablation lineage table (§18 output)

### 1.3 Per-attack-type write-up (§19 output)
- [x] 1.3.1 Replace `Attack_1`…`Attack_11` with real MDC label names (`mdc_label_map.py`)
- [x] 1.3.2 Report window counts per label (reference table in thesis draft)
- [x] 1.3.3 Explain sparse labels (only 0,1,2,3,4,8,11 in test set)
- [ ] 1.3.4 Re-run §19 + export named `eval_per_attack_type.png` at 300 dpi

### 3.2 Ablation study — §4.4
- [x] 3.2.1 Ablation harness (`no_contractive`, `no_early_stop`) with deltas vs default
- [ ] 3.2.2 Execute `RUN_ABLATIONS=True` on Colab GPU → `metrics_ablation_vnext.json`
- [x] 3.2.3 Split Table A/B + thesis paragraphs in `thesis_results_discussion.md`

### 1.4 Drift detection write-up (§14 output)
- [ ] 1.4.1 Report PSI major=99, moderate=30, KS median=0.62
- [ ] 1.4.2 Explain: holdout ≠ benign-only train by design
- [ ] 1.4.3 State clearly: **detection implemented, adaptation not yet**

---

## 2.0 PRIORITY 2 — Proposal Gaps (Drift-Aware System)

**Goal:** Close the gap between “drift monitor” and “drift-aware approach” in the proposal.  
**Effort:** 3–5 days coding + 1 run.  
**Where to add:** New sections in `mdc_model_vNext.ipynb` OR `notebook/mdc_drift_aware.ipynb` (recommended: separate notebook loading saved checkpoint).

### 2.1 Adaptive thresholding — §3.2 (REQUIRED for proposal)

**Add:** `adaptive_threshold(recent_benign_scores, alpha=2.5)`

```
threshold(t) = median(recent_benign) + alpha * std(recent_benign)
buffer_size = 100 windows, update every 20 windows
```

| Task | Deliverable |
|------|-------------|
| 2.1.1 | Implement `adaptive_threshold()` function |
| 2.1.2 | Simulate on test set in time order (chronological windows) |
| 2.1.3 | Compare FPR: fixed (f1_optimal) vs adaptive — first 50% vs last 50% of stream |
| 2.1.4 | Plot: threshold over time + FPR over time (WBS §5.3.2) |
| 2.1.5 | Record: adaptive FPR reduction (or document if no improvement) |

**Success criterion:** Adaptive FPR on last 50% ≤ fixed FPR, OR honest negative result in thesis.

### 2.2 Incremental fine-tune — §3.3 (REQUIRED for proposal)

**Add:** `fine_tune_on_recent(model, benign_windows, steps=50, lr=1e-5)`

| Task | Deliverable |
|------|-------------|
| 2.2.1 | Load `checkpoint_vnext.pt` from `runs/vnext_run/` |
| 2.2.2 | Split test chronologically: first 70% = stable, last 30% = drift period |
| 2.2.3 | Measure ROC-AUC on last 30% **before** fine-tune |
| 2.2.4 | Trigger when mean PSI > 0.10 (or use last-30% as forced drift simulation) |
| 2.2.5 | Fine-tune on 200 recent **benign** windows from drift period |
| 2.2.6 | Measure ROC-AUC on last 30% **after** fine-tune |
| 2.2.7 | Bar chart: AUC before vs after (WBS §5.3.4) |

**Success criterion:** AUC_after ≥ AUC_before on drift slice, OR document as limitation.

### 2.3 Drift-aware orchestration — §3.4 (NEW — ties 2.1 + 2.2)

**Add:** Single `drift_aware_eval()` loop (can live in `notebook/mdc_drift_aware.ipynb`)

| Step | Action |
|------|--------|
| 1 | Score window |
| 2 | Update benign score buffer |
| 3 | Every 20 windows: update adaptive threshold |
| 4 | Every 100 windows: compute PSI vs `drift_baseline_vnext.npz` |
| 5 | If PSI > 0.10: call `fine_tune_on_recent()` once |
| 6 | Log: score, threshold, alert, PSI, ground truth |

| Task | Deliverable |
|------|-------------|
| 2.3.1 | Implement loop over chronological test windows |
| 2.3.2 | Save `drift_aware_log.json` + `drift_aware_log.csv` |
| 2.3.3 | PSI timeline plot (WBS §5.3.3) |
| 2.3.4 | Score timeline with attack shading (WBS §5.3.1) |

---

## 3.0 PRIORITY 3 — Evaluation Completeness (Proposal Rigor)

**Goal:** Comparisons and ablations expected in a research thesis.  
**Effort:** 2–4 days (each ablation ~30 min GPU).

### 3.1 Baseline comparison — §4.3

| Model | Notebook section | Status |
|-------|------------------|--------|
| Isolation Forest | New cell or `mdc_baselines.ipynb` | **TODO** |
| Dense AE (optional) | Same | **TODO** |
| vNext HPO (yours) | Done | **0.8138 ROC** |

| Task | Deliverable |
|------|-------------|
| 3.1.1 | Flatten windows (N, T×F), fit `IsolationForest(contamination=0.38)` |
| 3.1.2 | Score test set, same metrics at f1_optimal |
| 3.1.3 | Comparison table: IF vs vNext default vs vNext HPO |

### 3.2 Ablation study — §4.4

| Config | Change | Status |
|--------|--------|--------|
| Full vNext HPO | — | **0.8138** (done) |
| No contractive | `contractive_lambda=0` | TODO |
| No early AUC stop | train to MAX_EPOCH | TODO |
| AUC-best only (no ensemble) | already default | **0.7402** (done) |
| Fixed vs adaptive threshold | from §2.1 | TODO |
| No fine-tune on drift | from §2.2 | TODO |

| Task | Deliverable |
|------|-------------|
| 3.2.1 | Run 2–3 ablations with seed=42 |
| 3.2.2 | Ablation table with delta vs full model |
| 3.2.3 | One paragraph per ablation in thesis |

### 3.3 Statistical rigor (lightweight)

| Task | Deliverable |
|------|-------------|
| 3.3.1 | Bootstrap 95% CI on test ROC/MCC (1000 resamples) |
| 3.3.2 | TPR @ FPR = 5%, 10%, 25% from saved scores |
| 3.3.3 | Inference latency: ms/window on CPU (WBS §5.2) |

---

## 4.0 PRIORITY 4 — Streaming Showcase (Viva / Demo)

**Goal:** Visual proof of “real-time drift-aware detection”.  
**Effort:** 2–3 days.

### 4.1 Notebook demo (minimum)
**File:** `notebook/mdc_drift_aware.ipynb` (recommended)

| Cell | Content |
|------|---------|
| 1 | Load checkpoint + npz + drift baseline |
| 2 | Streaming simulation loop (§2.3) |
| 3 | Plot 1 — score timeline |
| 4 | Plot 2 — adaptive vs fixed threshold |
| 5 | Plot 3 — PSI over time |
| 6 | Plot 4 — AUC before/after fine-tune |

### 4.2 Streamlit dashboard (optional)
**File:** `dashboard/module4_demo.py`

- Page 1: Live stream playback
- Page 2: Drift monitor + fine-tune trigger
- Page 3: Evaluation summary

### 4.3 Latency benchmark — §5.2

```
Target: < 100 ms/window on CPU
Record: windows/second
```

---

## 5.0 PRIORITY 5 — Thesis Deliverables

### 5.1 Required figures (300 dpi)

| # | Figure | Source |
|---|--------|--------|
| 1 | Architecture diagram | Draw from model §5 |
| 2 | Preprocess flowchart | From preprocess vNext |
| 3 | Training AUC/loss curve | `mdc_model_vNext_lat_output` §7 |
| 4 | Score distribution | §12 output |
| 5 | ROC + PR curves | §12 output |
| 6 | Score timeline (streaming) | §2.3 / §4.1 |
| 7 | Adaptive threshold plot | §2.1 |
| 8 | PSI drift signal | §2.3 |
| 9 | Per-attack recall bar | §19 output |
| 10 | AUC before/after fine-tune | §2.2 |
| 11 | Ablation table | §3.2 |
| 12 | Baseline comparison | §3.1 |

### 5.2 Results template (fill from latest run)

```
Model       : SequenceBottleneckAE vNext (HPO)
Dataset     : MDC (Misuse Detection in Containers)
Windows     : T=10, STRIDE=2, 163 features, 38.2% attack rate

Test (default, f1_optimal):
  ROC-AUC  : 0.7402
  MCC      : 0.5884
  F1       : 0.7412
  FPR      : 0.1403
  FNR      : 0.2774

Multi-seed (mean ± std):
  ROC-AUC  : 0.7529 ± 0.0231
  MCC      : 0.5771 ± 0.0405

HPO best:
  ROC-AUC  : 0.8138
  MCC      : 0.6260
  F1       : 0.7736

Drift monitor:
  PSI major (>0.25) : 99 / 163 features
  KS median         : 0.6166

Per-attack (top):
  Attack_1 recall   : 0.843 (n=1679)
  Attack_2 recall   : 0.023 (n=266)
  Benign FPR        : 0.140 (n=3184)

Adaptive threshold  : [FILL after §2.1]
Fine-tune AUC delta : [FILL after §2.2]
Isolation Forest    : [FILL after §3.1]
Inference latency   : [FILL after §3.3]
```

---

## 6.0 Recommended File Plan (What to Add)

| File | Purpose | Priority |
|------|---------|----------|
| `notebook/mdc_drift_aware.ipynb` | Adaptive threshold + fine-tune + streaming loop | **P2 — Required** |
| `notebook/mdc_baselines.ipynb` | Isolation Forest (+ optional Dense AE) | **P3** |
| `dashboard/module4_demo.py` | Viva Streamlit demo | P4 optional |
| `docs/Module4_WBS.md` | This document | Done |
| ~~`notebook/mdc_eval_multiclass.ipynb`~~ | Removed — merged into model §19 | N/A |

**Do NOT create new preprocess/model pipelines** — extend from saved checkpoint.

---

## 7.0 Execution Order (Updated)

```
PHASE A — DONE (cite in thesis, no rework)
  ✓ Preprocess vNext + vNext_mc
  ✓ Model train + HPO + multi-seed
  ✓ Drift baseline + PSI/KS monitor
  ✓ Per-attack eval §19
  ✓ Artifacts saved (local cache + Drive copy)

PHASE B — Thesis write-up (1–2 days, no code)
  → 1.0 Fill tables, figures from lat_output
  → Map attack label names
  → Document score flip + drift detection (not adaptation)

PHASE C — Proposal gaps (3–5 days) *** CRITICAL FOR PROPOSAL ALIGNMENT ***
  → 2.1 Adaptive thresholding
  → 2.2 Incremental fine-tune demo
  → 2.3 Drift-aware streaming loop + 4 plots
  → New notebook: mdc_drift_aware.ipynb

PHASE D — Research rigor (2–4 days)
  → 3.1 Isolation Forest baseline
  → 3.2 2–3 ablations (contractive, early stop)
  → 3.3 Bootstrap CI + TPR@FPR + latency

PHASE E — Showcase (1–3 days, before viva)
  → 4.1 Drift-aware demo notebook
  → 4.2 Streamlit (optional)
  → 5.1 Export all figures 300 dpi
```

---

## 8.0 Blockers and Fallbacks

| Blocker | Fallback |
|---------|----------|
| Fine-tune does not improve AUC | Report as negative result; proposal claim → "future work" |
| Adaptive threshold no FPR gain | Report fixed threshold as primary; adaptive as attempted |
| No time for Streamlit | Use `mdc_drift_aware.ipynb` cell-by-cell at viva |
| No time for Dense AE baseline | Isolation Forest only (minimum) |
| Colab session loss | Use `checkpoint_vnext.pt` + gdown folder — never retrain from scratch |

---

## 9.0 Honest Proposal Positioning (for thesis text)

**What you can claim today:**
- End-to-end anomaly detection pipeline with strong metrics (0.81 HPO ROC)
- Leakage-free preprocessing with multiclass evaluation
- Drift **monitoring** infrastructure (baseline + PSI/KS)

**What you must implement to claim "drift-aware":**
- Adaptive threshold (§2.1)
- Incremental fine-tune with before/after evidence (§2.2)
- Streaming simulation tying them together (§2.3)

**What you can label "future work" if time runs out:**
- Production deployment
- Cross-dataset generalization
- Automated full retrain pipeline (not just fine-tune)

---

*Document version: 2026-07-08 (post vNext lat run)*  
*Module 4 — Security Anomaly Detector — DracaSys Group 24*
*University of Moratuwa, Faculty of Information Technology*
