# 04 — Final Model, Results & Reproducibility

> Part of a 5-file documentation set. This file is the authoritative results record —
> every number below is read directly from
> `final_notebook/output-metrics/kagglephase3_output/{all_metrics.json,
> all_metrics_summary.csv, all_metrics_per_target.csv}` and
> `final_notebook/output-metrics/kagglephase4_output/phase4_results.json`, and
> cross-checked against the executed notebook cell outputs in
> `final_notebook/final-output/`. Where a number could not be independently verified,
> it is marked `Not found / Not verified` rather than assumed.

---

## 1. Final Architecture

**`AdaptiveGRUModel`** (defined `phase2 final.ipynb`, Step 2; source embedded in
`output-metrics/kagglephase2_defs/model_defs.py`):

```text
Input:  (batch, variable_length in [500,1000], 27 features), packed sequence
GRU:    2 layers, hidden_size=128, dropout=0.2 between layers
Head:   Dropout -> Linear(128, 64) -> ReLU -> Linear(64, 4)
Output: correction (batch, 4)
Final:  prediction = correction + x[:, last_valid_step, residual_target_indices]
        (residual_indices = [0, 3, 4, 5] -- the same 4 columns are both input
        features and prediction targets)
Init:   fc2 (final layer) weights and bias zero-initialized when residual mode is
        on, so at epoch 0 the model outputs exactly the persistence prediction
```

**Verified parameter count** (every checkpoint's stored state, and the Phase 2 smoke
test): **167,876 trainable parameters**, identical across all 3 trained horizons.

**Why residual/persistence-anchored, not a plain regression head:** documented in
`03_experiments_and_research_history.md` §"V8" — every prior raw-value formulation
lost to a trivial "predict no change" baseline; anchoring the output to the last
observed value gives the model a structural floor at persistence performance instead
of an accidental one.

---

## 2. Training Process

**Config** (`all_metrics.json` → `config`, verified identical to `phase3
final.ipynb` Step 2's printed `Phase3Config`):

| Parameter | Value |
|---|---|
| horizons | [1, 2, 3] (15s, 30s, 45s ahead) |
| hidden_size | 128 |
| num_layers | 2 |
| dropout | 0.2 |
| batch_size (train) | 128 |
| eval_batch_size | 512 |
| optimizer | Adam, lr=0.001 |
| max epochs | 30 |
| early-stopping patience | 7 |
| scheduler | ReduceLROnPlateau (implementation detail in `model_defs.py`'s `train_model`) |

**Loss function:** MSE, computed in normalized (z-score) space.

**Checkpointing:** best-validation-loss only (see `05_model_integration_and_developer_guide.md`
§1 for the exact save mechanism).

**Verified per-horizon training outcome** (`checkpoints` block in `all_metrics.json`):

| Horizon | Best epoch | Best val_loss (normalized MSE) | Checkpoint size |
|---|---|---|---|
| 1 | 6 | 0.000329 | 0.68 MB |
| 2 | 7 | 0.000631 | 0.68 MB |
| 3 | 21 | 0.001012 | 0.68 MB |

Horizon 3 needed substantially more epochs (21 vs. 6–7) to reach its best checkpoint —
consistent with longer-horizon targets being a harder learning problem, as expected.

**Hyperparameter optimization:** `Not found / Not verified`. No grid search, random
search, or Bayesian optimization artifacts exist anywhere in `module2/`. The config
above appears to be a fixed choice carried forward from the `outputs-legacy-experiments/`
iterations, not the product of a documented search.

**Multi-seed evaluation:** present in the historical `outputs-legacy-experiments/`
line (e.g. `v8_kaggle_final_output.ipynb`'s multi-seed variance check), **absent** in
the final `final_notebook/final-output/` 4-notebook pipeline. The final reported
results below are single-seed point estimates — no standard deviation or confidence
interval is available for any of them.

---

## 3. Final Metrics

**Important scope note before any table:** this is a regression/forecasting task.
Classification-style metrics (ROC-AUC, F1, MCC, Precision, Recall, Accuracy, PR-AUC,
confusion matrix) requested by the general documentation brief are **Not applicable**
— nowhere in this project is there a binary/multiclass anomaly decision. The
applicable metrics, consistently used throughout, are **MAE, RMSE, and MAPE**, all
computed in de-normalized real units (CPU-seconds for cpu_usage, bytes for the 3
memory targets) after inverse z-scoring.

### 3.1 Static evaluation — CLEAN (untouched) test data

Source: `all_metrics_per_target.csv`, `variant=clean, method=GRU-static`, cross-checked
against `phase3 final.ipynb` cell 12's printed "TABLE A".

| Horizon | Target | Persistence MAPE | SES MAPE | GRU-static MAPE |
|---|---|---|---|---|
| 1 (15s) | cpu_usage | 0.014% | 0.014% | 0.044% |
| 1 | mem_usage | 0.066% | 0.073% | 0.093% |
| 1 | mem_working_set | 0.066% | 0.078% | 0.093% |
| 1 | mem_rss | 0.074% | 0.087% | 0.099% |
| 2 (30s) | cpu_usage | 0.027% | 0.027% | 0.213% |
| 2 | mem_usage | 0.132% | 0.143% | 0.203% |
| 2 | mem_working_set | 0.132% | 0.143% | 0.203% |
| 2 | mem_rss | 0.148% | 0.160% | 0.219% |
| 3 (45s) | cpu_usage | 0.041% | 0.041% | 0.146% |
| 3 | mem_usage | 0.198% | 0.210% | 0.217% |
| 3 | mem_working_set | 0.198% | 0.211% | 0.219% |
| 3 | mem_rss | 0.222% | 0.230% | 0.242% |

**On clean, real data, persistence and SES beat GRU-static on every single
target/horizon combination shown above (12/12).** This is the same persistence-
dominance pattern first found in V4 (`03_experiments_and_research_history.md`) and
never fully overcome — reported honestly here, not adjusted.

### 3.2 Static evaluation — BURST-INJECTED test data (batch, no streaming/adaptation)

Source: `all_metrics_per_target.csv`, `variant=injected`.

| Horizon | Target | Persistence MAPE | SES MAPE | GRU-static MAPE | GRU beats both? |
|---|---|---|---|---|---|
| 1 | cpu_usage | 0.027% | 0.027% | 0.031% | No |
| 1 | mem_usage | 0.166% | 0.184% | 0.164% | Yes (marginal) |
| 1 | mem_working_set | 0.166% | 0.184% | 0.164% | Yes (marginal) |
| 1 | mem_rss | 0.172% | 0.191% | 0.168% | Yes |
| 2 | cpu_usage | 0.053% | 0.053% | 0.102% | No |
| 2 | mem_usage | 0.330% | 0.339% | 0.247% | Yes |
| 2 | mem_working_set | 0.330% | 0.348% | 0.246% | Yes |
| 2 | mem_rss | 0.343% | 0.361% | 0.256% | Yes |
| 3 | cpu_usage | 0.080% | 0.080% | 0.091% | No |
| 3 | mem_usage | 0.494% | 0.519% | 0.311% | Yes |
| 3 | mem_working_set | 0.495% | 0.519% | 0.313% | Yes |
| 3 | mem_rss | 0.513% | 0.529% | 0.333% | Yes |

**Precise, non-oversimplified finding:** under burst injection, GRU-static beats both
baselines on **all 3 memory targets at every horizon** (9/9), but **loses to
persistence on cpu_usage at every horizon** (0/3) — though the margin is small
(0.03–0.09 percentage points). Reporting this at the aggregate (4-target mean) level,
as Phase 4's streaming results do, correctly shows GRU ahead overall (§3.3), but that
aggregate is being pulled by the memory targets; the per-target reality for cpu_usage
specifically remains a near-tie leaning toward persistence even under burst
conditions. **Do not cite an unqualified "GRU beats baselines under bursts" claim
without this caveat** — the per-target table above is the evidence that should
accompany any such claim.

### 3.3 Spike-regime vs. normal-regime split (still batch/static, `phase3 final.ipynb` Step 5)

Source: `all_metrics_per_target.csv`, `variant ∈ {injected_spike, injected_normal}`.

| Horizon | Regime | cpu_usage | mem_usage | mem_working_set | mem_rss |
|---|---|---|---|---|---|
| 1 | spike (10,028 windows) | 0.068% | 0.293% | 0.293% | 0.294% |
| 1 | normal (23,315 windows) | 0.015% | 0.109% | 0.108% | 0.114% |
| 2 | spike (10,119 windows) | 0.169% | 0.416% | 0.415% | 0.410% |
| 2 | normal (23,224 windows) | 0.073% | 0.173% | 0.173% | 0.189% |
| 3 | spike (10,210 windows) | 0.194% | 0.546% | 0.547% | 0.564% |
| 3 | normal (23,133 windows) | 0.046% | 0.208% | 0.209% | 0.232% |

Error is consistently 2–4× higher inside spike-affected windows than normal windows,
for every target and horizon — the burst events are measurably harder to predict, as
intended by the experiment design.

### 3.4 Streaming, drift-aware evaluation — Static vs. Adaptive (`phase 4.ipynb`)

Source: `phase4_results.json` → `stream_results` (aggregate MAPE, mean across all 4
targets), cross-checked against `phase 4.ipynb` cells 6/8's printed output.

| Horizon | Static overall | Adaptive overall | Static spike | Adaptive spike | Static normal | Adaptive normal | Online updates |
|---|---|---|---|---|---|---|---|
| 1 | 0.132% | 0.118% | 0.237% | 0.227% | 0.086% | 0.071% | 3 (chunks 11,13,15) |
| 2 | 0.213% | 0.200% | 0.353% | 0.341% | 0.152% | 0.139% | 9 (chunks 10,12,14,16,21,23,25,27,32) |
| 3 | 0.262% | 0.262% | 0.463% | 0.449% | 0.174% | 0.180% | 9 (chunks 10,12,14,16,20,22,24,26,28) |

**Adaptation improves spike-window MAPE at all 3 horizons** (0.237→0.227,
0.353→0.341, 0.463→0.449) and normal-window MAPE at 2 of 3 (horizon 3's normal
windows moved slightly the wrong way, 0.174→0.180 — a small, honestly-reported
regression, not hidden). Overall aggregate MAPE improved or held flat at every
horizon. Magnitude of improvement is modest (0.01–0.02 percentage points on overall
MAPE) — the mechanism is demonstrably real and working (§4 below), but its net
accuracy benefit on this dataset is small, not dramatic.

**Drift-detection mechanism, direct proof (not inferred):** the horizon-3 per-chunk
debug log (`phase4_results.json` → `debug_log`, 44 rows, also printed in `phase
4.ipynb` cell 10) shows: warmup (chunks 0–7) establishes a reference error mean of
0.000645 with std 0.000107; chunk 9's error jumps to z-score 5.05 (first hit); chunk
10 reaches z-score 17.09 (second consecutive hit) → triggers, with a verified
parameter-checksum change confirming the update genuinely modified the model
(`weights_changed: YES` on every one of the 9 recorded triggers). After the last
trigger (chunk 28), the z-score falls monotonically to −2.86 by the stream's end, and
0 of the remaining 15 chunks exceed the trigger threshold — the notebook's own
automated diagnosis (`phase 4.ipynb` cell 14) concludes: *"error genuinely stabilized
well below the reference threshold... the mechanism worked as intended."*

---

## 4. Proposal Novelty Components — Verified Implementation Status

| Component | Status | Evidence |
|---|---|---|
| Online/incremental learning, error-triggered updates | **Implemented, verified working** | `OnlineAdapter` (model_defs.py), fires 3–9 times per horizon in Phase 4, every trigger's weight-change checksum confirmed non-zero |
| Error monitoring (moving average + statistical checks) for drift | **Implemented, verified working — strongest evidence of the three** | `DriftMonitor` (EWMA + z-score vs. frozen warmup reference), full causal record in the per-chunk debug log above |
| Adaptive thresholding | **Implemented, executes live, but its output is not surfaced or analyzed anywhere** | `AdaptiveThreshold` computed every chunk (`phase 4.ipynb` cell 6/10), values persisted in `debug_log`'s `athresh_lo`/`athresh_hi` fields, but never printed in a summary table, plotted, or used to gate any decision in any notebook |
| Adaptive sliding window | **Implemented as a historically-precomputed variable length, not live at inference time** | `02_data_pipeline_and_methodology.md` §7 — 51.6% of train windows measurably shortened (real effect), but the length rule runs once in Phase 1 from training-period statistics and is never recomputed during Phase 3/4 evaluation |
| Burst-aware forecasting | **Implemented and evidenced, with the honest caveat in §3.2** | Synthetic burst injection (Phase 1) + spike/normal regime split (Phase 3/4); aggregate GRU advantage confirmed, but driven by memory targets, not cpu_usage |
| Multi-metric correlation modeling | **Not implemented as a distinct mechanism** | The model shares one GRU hidden state across 4 output heads (an incidental joint representation), but there is no explicit correlation term, joint loss, or cross-metric attention anywhere in `model_defs.py`. Network/disk metrics are also entirely absent (§1 of `02_data_pipeline_and_methodology.md`) |

---

## 5. Baselines

| Baseline | Method | Where computed |
|---|---|---|
| Persistence | `prediction(t+h) = value(t)` — the last observed value, no training | `phase3 final.ipynb` Step 4, `persistence_preds()` in `model_defs.py` |
| Simple Exponential Smoothing (SES) | Weighted average of the lookback window, alpha fit per-target on the train split only (grid search over `[0.05, 1.0]` step 0.05, minimizing 1-step MSE) | `phase3 final.ipynb` Step 4, `ses_fit_alpha()`/`ses_preds()` in `model_defs.py` |

**Classical ML baselines (RandomForest, Isolation Forest, XGBoost, etc.):**
`Not present`. Only the two non-learned baselines above and the GRU family exist in
this project.

---

## 6. Exact Reproduction Procedure

```text
Prerequisite: a Kaggle account with GPU quota, and the raw AIOpsArena
complex_case1 CSVs uploaded as a Kaggle Dataset (e.g. "raw-data").

Step 1 → Open final_notebook/final-source/phase1 final.ipynb in Kaggle
         Input:  attach the "raw-data" Dataset via Add Input
         Run:    Run All  (~5 minutes, GPU not required)
         Output: download kagglephase1_output.zip from the Output tab

Step 2 → Upload the unzipped contents of kagglephase1_output.zip as a new
         Kaggle Dataset named "kagglephase1-output"

Step 3 → Open final_notebook/final-source/phase2 final.ipynb in Kaggle
         Input:  attach "kagglephase1-output" (optional, enables the real-file
                 smoke-test check)
         Run:    Run All  (~2 minutes)
         Output: download kagglephase2_defs.zip, upload as Dataset "kagglephase2-defs"

Step 4 → Open final_notebook/final-source/phase3 final.ipynb in Kaggle
         Input:  attach "kagglephase1-output" AND "kagglephase2-defs"
         Run:    Run All  (~40-70 minutes, GPU required, 3 horizons)
         Output: download kagglephase3_output.zip, upload as Dataset "kagglephase3-output"

Step 5 → Open final_notebook/final-source/phase4 final.ipynb in Kaggle
         Input:  attach "kagglephase1-output", "kagglephase2-defs", AND
                 "kagglephase3-output"
         Run:    Run All  (~5-10 minutes)
         Output: kagglephase4_output.zip (streaming results + debug log + plot)
```

**Randomness / seeding:** `set_seed()` (`model_defs.py`) sets `random`, `numpy`, and
`torch` seeds per horizon before model construction (verified present in
`train_all_horizons`-equivalent training code called from `phase3 final.ipynb` Step
3). The synthetic burst injection uses 3 fixed seeds (train=101, val=202, test=303,
`manifest.json`). **The exact seed value(s) used for the final reported training runs
are `Not found / Not verified` from the saved artifacts alone** — the per-horizon seed
convention is documented in code but the specific integers used for this exact run
are not recorded in `all_metrics.json` or the checkpoints.

**Non-reproducibility risk, disclosed directly:** no `requirements.txt` exists;
the only recorded dependency version is `PyTorch 2.10.0+cu128` (from a printed cell
output, not a pinned file) — a different PyTorch/CUDA build in a future Kaggle session
could reproduce different numeric results, and this project's own history includes a
CUDA kernel-mismatch failure caused by exactly this kind of environment drift (see
`05_model_integration_and_developer_guide.md` §6, Troubleshooting).
