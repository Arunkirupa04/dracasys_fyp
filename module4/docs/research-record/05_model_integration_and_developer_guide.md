# 05 — Model Integration and Developer Guide

**Audience:** Developers integrating the frozen Module 4 detector into an external application.  
**Status of serving code in-repo:** **Not found** — no FastAPI/Flask/Docker service under `module4/`. Sections marked **Proposed** are technically compatible designs derived from the real checkpoint/score contract; they are not implemented project code.  
**Canonical artifacts:** `module4/notebook/final/output-metrics/`  
**Model source of truth:** `module4/notebook/final/kaggle-source/mdc_model_vNext_kaggle.ipynb`  
**Related:** [02 data/preprocess](02_data_pipeline_and_methodology.md) · [04 metrics](04_final_model_results_and_reproducibility.md)

---

## 1. Required components

| Component | Path (freeze) | Required? | Notes |
|-----------|---------------|-----------|-------|
| Model checkpoint | `model_vnext_runs/checkpoint_vnext.pt` | **Yes** | PyTorch `torch.save` dict |
| Metrics / threshold | `model_vnext_runs/metrics_vnext.json` | **Yes** | `primary_threshold`, `auto_score_flip`, `test_default.f1_optimal.threshold` |
| Feature std (optional) | Inside checkpoint key `feat_std` | If non-null | Applied in scoring when present |
| Architecture config | Checkpoint key `config` | **Yes** | Rebuild `SequenceBottleneckAE` |
| Pre-scaled windows contract | `windows_vnext.npz` shape/schema | For offline replay | Already scaled |
| Preprocess pickle | `windows_vnext_processed/preproc_vnext.pkl` | For live raw flows | **Provenance**; not loaded by current notebooks |
| Label map | `mdc_label_map.json` | Optional | Multiclass naming only |
| Drift adaptive artifacts | `drift_aware_outputs/*` | Optional | Research; **not** production decision layer in this freeze |

### 1.1 Dependencies (runtime)

| Dependency | Role | Pin status |
|------------|------|------------|
| Python 3.10+ (freeze ran 3.11–3.12) | Runtime | Unpinned |
| PyTorch | Load AE / forward | **Not pinned** in-repo |
| NumPy | Arrays | Unpinned |
| scikit-learn / joblib | Only if replaying preprocess pickle | Unpinned |
| FastAPI / uvicorn | **Proposed** API only | Not in project |

Exact torch/sklearn versions used on Kaggle for the freeze: **Not verified**.

### 1.2 Hardware

- Inference: CPU is sufficient at ~2.5 ms median latency (`stats_report.json`).
- Training / HPO: GPU recommended (proposal: RTX 3060-class). Not required for score-only integration.

---

## 2. What is saved in the checkpoint

**Filename:** `checkpoint_vnext.pt`  
**Save site:** Model notebook cell `cell-save` → `/kaggle/working/runs/checkpoint_vnext.pt`  
**Load site (example):** Drift-aware notebook — `torch.load(..., weights_only=False)`

### 2.1 Top-level keys

| Key | Contents |
|-----|----------|
| `model_default` | `state_dict` of default `SequenceBottleneckAE` |
| `model_best` | `state_dict` of HPO-retrained model |
| `hpo_params` | Best Optuna parameter dict |
| `feat_std` | Per-feature std array or `None` |
| `config` | `{T, n_features, d_model, bottleneck_dim, nhead, num_enc_layers, num_dec_layers, dim_ff, dropout}` |

**Not stored in checkpoint:** `SCORE_INVERT`, score mean/max weights, thresholds, ensemble flag.  
Invert flag for the freeze: `metrics_vnext.json` → `"auto_score_flip": true` (and drift summary `score_invert_production: true`).  
Primary threshold: `metrics_vnext.json` → `test_default.f1_optimal.threshold` = **−0.19924625754356384**.

Exact `state_dict` key list from loading a `.pt` in this documentation pass: **Not verified** (inferred modules: `input_proj`, `pe`, `encoder`, `bn_down`, `bn_up`, `decoder`, `output_proj`, `pos_queries`).

### 2.2 Which weights to load

| Use case | Load key | Threshold source |
|----------|----------|------------------|
| Match drift-aware / default integrity baseline | `model_default` | `test_default.f1_optimal.threshold` |
| Best-case HPO detector | `model_best` | `test_hpo.threshold` (−0.3474787175655365) |

Drift-aware default config uses `USE_HPO_MODEL=False` because saved `scores_vnext.npz` were produced from the default model.

### 2.3 Other saved training artifacts

| File | Keys / meaning |
|------|----------------|
| `scores_vnext.npz` | `s_val`, `s_test`, `y_val`, `y_test` — offline scores |
| `metrics_vnext.json` | Full metric + threshold report |
| `preproc_vnext.pkl` | `scaler_flow`, `scaler_bucket`, masks, clip bounds, feature lists, window hyperparams (joblib) |

Baseline Dense AE / Isolation Forest: **no durable model file** in freeze (metrics JSON only).

---

## 3. Input and output schemas

### 3.1 Model tensor input

| Field | Spec |
|-------|------|
| Shape | `(B, T, F) = (batch, 10, 163)` |
| Dtype | `float32` |
| Semantics | 10 consecutive 15 s buckets; 163 scaled features per bucket |
| Scale | **Must match training scale** (dual StandardScaler + clip ±10 from preprocess) |

### 3.2 Binary decision output (research production)

| Field | Type | Meaning |
|-------|------|---------|
| `score` | float | Anomaly score (higher = more anomalous **after** invert, in freeze) |
| `threshold` | float | From metrics JSON |
| `is_anomaly` | bool | `score >= threshold` when `ALERT_INVERT=false` on saved score space |

Drift-aware uses `ALERT_INVERT=False` on already-inverted production scores.

### 3.3 Multiclass

The AE does **not** emit attack-type IDs. Multiclass labels exist only for evaluation (`y_*_multiclass`). Per-attack eval JSON for this freeze: **Not found**.

---

## 4. Inference process (actual project logic)

```text
Load checkpoint_vnext.pt
    ↓
Rebuild SequenceBottleneckAE from config
    ↓
Load state_dict (model_default or model_best)
    ↓
Load threshold (+ confirm auto_score_flip) from metrics_vnext.json
    ↓
Prepare input window tensor (B,10,163) float32 — already scaled
    ↓
model.eval(); recon = model(x)
    ↓
err = (recon - x)^2
optional: err = err / feat_std
    ↓
score = 0.3 * mean(err) + 0.7 * max_over_T(mean_over_F(err))
    ↓
if invert: score = -score
    ↓
alert = (score >= threshold)   # ALERT_INVERT=false convention
    ↓
Return {score, threshold, is_anomaly}
```

### 4.1 Score formula (code-faithful)

From model notebook scoring cell:

```text
err = (recon - x)²                         # (B, T, F)
# optional feat_std weighting on err
s_mean = mean(err over dims T,F)
s_max  = max_over_T( mean_over_F(err) )
score  = 0.3 * s_mean + 0.7 * s_max
score  = -score  if invert else score
```

Weights: `SCORE_MEAN_W=0.3`, `SCORE_MAX_W=0.7`.

### 4.2 Practical Python sketch (illustrative)

The following is a **developer sketch** aligned with artifact keys. It is not a checked-in module.

```python
import json
from pathlib import Path
import numpy as np
import torch

# NOTE: SequenceBottleneckAE class must be copied from
# module4/notebook/final/kaggle-source/mdc_model_vNext_kaggle.ipynb

ART = Path("module4/notebook/final/output-metrics")
ckpt = torch.load(ART / "model_vnext_runs/checkpoint_vnext.pt",
                  map_location="cpu", weights_only=False)
metrics = json.loads((ART / "model_vnext_runs/metrics_vnext.json").read_text())

cfg = ckpt["config"]
model = SequenceBottleneckAE(**{k: cfg[k] for k in [
    "T", "n_features", "d_model", "bottleneck_dim", "nhead",
    "num_enc_layers", "num_dec_layers", "dim_ff", "dropout"
]})  # constructor arg names must match notebook class
model.load_state_dict(ckpt["model_default"])
model.eval()

thr = metrics["test_default"]["f1_optimal"]["threshold"]
invert = bool(metrics.get("auto_score_flip", False))
feat_std = ckpt.get("feat_std")
MEAN_W, MAX_W = 0.3, 0.7

@torch.no_grad()
def score_windows(x: torch.Tensor) -> np.ndarray:
    # x: (B, 10, 163) float32
    recon = model(x)
    err = (recon - x) ** 2
    if feat_std is not None:
        fs = torch.as_tensor(feat_std, dtype=err.dtype, device=err.device)
        err = err / fs.clamp_min(1e-8).view(1, 1, -1)
    s_mean = err.mean(dim=(1, 2))
    s_max = err.mean(dim=2).amax(dim=1)
    s = MEAN_W * s_mean + MAX_W * s_max
    if invert:
        s = -s
    return s.cpu().numpy()

def predict(x: torch.Tensor):
    s = score_windows(x)
    return {"score": s, "threshold": thr, "is_anomaly": s >= thr}
```

**Caveats:**

1. Constructor keyword names must match the notebook class exactly — verify before production use.
2. If integrating **raw flows**, you must reproduce preprocess (or load `preproc_vnext.pkl`) — see §5.
3. Ensemble scoring path (rank-average of two checkpoints) may exist in the notebook; whether it produced the frozen `scores_vnext.npz`: **Not verified**. Prefer matching `scores_vnext.npz` on a held-out batch as a conformance test.

---

## 5. Required preprocessing before inference

### 5.1 Path A — Offline / research (what the project actually does)

```text
Run preprocess notebook → windows_vnext.npz (already scaled)
    ↓
Load windows into model
```

Downstream notebooks **never** call `joblib.load(preproc_vnext.pkl)`.

### 5.2 Path B — Live raw flows (**proposed**)

```text
Raw CIC-like flow records
    ↓
Sessionize (60 s gap), filter containers
    ↓
Apply saved variance/corr masks + clip bounds from preproc_vnext.pkl
    ↓
Transform with scaler_flow → bucket 15 s (mean/max/std + flow_count)
    ↓
Gap fill ≤4 → transform with scaler_bucket → clip ±10
    ↓
Emit sliding windows (T=10, stride=2) shape (N,10,163)
    ↓
Score with checkpoint
```

**Risk:** Any column-order or feature-set drift vs training will silently invalidate scores. Conformance-test against a known `windows_vnext.npz` batch.

---

## 6. Thresholding policy for integration

| Policy | Use in external app? | Evidence |
|--------|----------------------|----------|
| Fixed `f1_optimal` from metrics JSON | **Yes — research production default** | Integrity-gated; matches offline metrics |
| HPO threshold with `model_best` | Yes, if you deliberately ship HPO | Separate operating point |
| Adaptive median+α·std buffer | **Not recommended as default** | Final freeze `claim_ok=false`; last-half F1/recall → 0 |
| Recompute threshold online from mixed traffic | **Dangerous** | Caused v1 double-invert invalid FPR story |

---

## 7. External integration patterns (**proposed**)

`system_integration_architecture.md` is explicitly a **planning document**. No orchestrator envelope is implemented in code.

### 7.1 Conceptual flow

```text
External Application
        ↓
Container / system metrics (CIC-like flows in this project)
        ↓
Preprocessing (Path B)
        ↓
SequenceBottleneckAE
        ↓
Anomaly score
        ↓
Fixed threshold (f1_optimal)
        ↓
Detection result → alert sink / SIEM
```

### 7.2 In-process Python application

- Vendor the AE class + load checkpoint at process start.
- Batch windows for throughput; single-window latency ~ms on CPU.
- Log `score`, `threshold`, model version (`metrics.created_at` / file hash).

### 7.3 REST / FastAPI service (**proposed; not implemented**)

Example contract:

```http
POST /v1/score
Content-Type: application/json

{
  "windows": [[[...163 floats...], ... 10 timesteps ...]],
  "model": "default"
}
```

```json
{
  "scores": [0.12],
  "threshold": -0.19924625754356384,
  "is_anomaly": [true],
  "model": "default",
  "invert": true
}
```

Sketch:

```python
# Proposed only — not in repository
from fastapi import FastAPI
app = FastAPI()

@app.post("/v1/score")
def score(payload: dict):
    x = torch.tensor(payload["windows"], dtype=torch.float32)
    key = "model_best" if payload.get("model") == "hpo" else "model_default"
    # load corresponding weights + threshold...
    return predict(x)
```

### 7.4 Containerised service (**proposed**)

```text
Docker image:
  - python + torch CPU/GPU
  - checkpoint_vnext.pt
  - metrics_vnext.json
  - (optional) preproc_vnext.pkl + preprocess code
  - FastAPI entrypoint
Kubernetes:
  - Deployment + Service
  - Readiness: GET /healthz after torch.load success
```

No Dockerfile exists in `module4/` — **Not found**.

### 7.5 Monitoring pipeline (**proposed**)

```text
Flow exporter → message queue → window builder → scorer → alert bus
                     ↑
              PSI monitor (optional research component)
```

Use PSI / adaptive threshold only as **monitoring signals**, not as silent production decision flips, unless you re-validate claim gates on your traffic.

---

## 8. Conformance tests (recommended before shipping)

1. Load freeze `scores_vnext.npz` and re-score the same `X_test` windows; compare correlation / MAE of scores.
2. Apply `f1_optimal` thr; confirm confusion counts match `metrics_vnext.json` (TP=1427, FP=927, TN=2258, FN=541 for default).
3. Confirm input shape `(N,10,163)` and `n_features` in checkpoint config equals 163.
4. Never assert adaptive FPR reduction without matched-policy recall check.

---

## 9. Troubleshooting and common mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Forget score invert | AUC ≪ 0.5 or swapped classes | Respect `auto_score_flip` / production invert |
| Recompute thr with wrong invert (v1 bug) | Stream FPR ~0.97; fake “FPR↓0.92” | Load thr from metrics JSON; do not double-invert |
| Use adaptive thr as default | Zero alerts / recall 0 on drifting half | Keep fixed `f1_optimal` |
| Cite `freeze_manifest` locked 0.7402 as this freeze’s ROC | Wrong number | Cite `metrics_vnext.json` (0.6889 default) |
| Load `preproc_vnext.pkl` onto already-scaled NPZ | Double scaling | Scale raw flows **or** use NPZ, not both |
| Confuse buffer size 500 with window length | Wrong input shape | Window T=10; buffer 500 is adaptive benign buffer |
| Ship `model_best` with default thr | Miscalibrated alerts | Pair HPO weights with `test_hpo.threshold` |
| Expect CPU/mem/syscall features | Missing columns | Freeze is flow-only (163 CIC-style features) |
| Expect live Falco integration | Not present | Planning doc only |

---

## 10. Known limitations (integration view)

1. Notebook-only research codebase — no maintained SDK.
2. Preprocess pickle not wired into inference notebooks.
3. Adaptive / fine-tune layers are research evaluations, not proven production improvements in this freeze.
4. No multiclass attack-type head.
5. Dependency versions unpinned.
6. Proposal multi-modal telemetry not available in frozen features.
7. Standalone API/Docker/K8s: **Not found / Not implemented**.

---

## 11. Artifact quick map for handoff

```text
Hand to another developer:
  1. checkpoint_vnext.pt
  2. metrics_vnext.json
  3. SequenceBottleneckAE class source (from mdc_model_vNext_kaggle.ipynb)
  4. (If live) preproc_vnext.pkl + preprocess notebook as specification
  5. This file + 02 (preprocess contract) + 04 (expected metrics for conformance)
```

---

## 12. Not found / Not verified

| Item | Status |
|------|--------|
| In-repo REST/FastAPI service | **Not found** |
| Dockerfile / Helm chart | **Not found** |
| `mdc_eval_utils.py` as standalone module | **Not found** (mentioned in older README; not present under module4) |
| Exact state_dict key listing from loaded `.pt` | **Not verified** |
| Whether frozen scores used ensemble ranks | **Not verified** |
| Whether `feat_std` is non-null in this checkpoint file | **Not verified** without loading `.pt` |
| Pinned torch/sklearn versions | **Not found** |
