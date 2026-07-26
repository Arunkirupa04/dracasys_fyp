# 05 — Model Integration & Developer Guide

> Part of a 5-file documentation set. For a developer who wants to load the trained
> model and use it — without reading the research history — this file is the entry
> point. Model files referenced here live in
> `final_notebook/output-metrics/kagglephase3_output/checkpoints/`.

---

## 1. Saved Model Structure

**Format:** PyTorch checkpoint (`torch.save`/`torch.load`), **not** pickle or joblib.
Pickle/joblib is the correct format for scikit-learn estimators; this project has no
scikit-learn models anywhere, so that format is not applicable here. `torch.save`
internally uses pickle for the container structure but adds device-mapping and
dedicated tensor storage that raw pickle lacks — see `model_defs.py`'s `train_model`
for the exact call site.

**Files** (verified present, `final_notebook/output-metrics/kagglephase3_output/checkpoints/`):

| File | Size | Best epoch | Best val_loss |
|---|---|---|---|
| `gru_h1_static.pt` | 0.68 MB | 6 | 0.000329 |
| `gru_h2_static.pt` | 0.68 MB | 7 | 0.000631 |
| `gru_h3_static.pt` | 0.68 MB | 21 | 0.001012 |

**One file per horizon** — there is no single "the model," there are 3 independent
models, one per forecast distance (15s/30s/45s ahead).

**Exact save call** (`model_defs.py`, inside `train_model`):

```python
torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch,
            'val_loss': vl}, ckpt_path)
```

Only saved **when validation loss improves** on the best seen so far — the file
always holds the best-generalizing weights, not the final training epoch's weights
(early stopping typically runs several epochs past the best one before halting).

**What is NOT inside the checkpoint** — read this before writing any loading code:

- **No architecture config.** `hidden_size`, `num_layers`, `dropout`, and critically
  `residual_indices` are not stored. You must supply the exact same constructor
  arguments used at training time.
- **No optimizer state.** These checkpoints cannot resume interrupted training with
  full fidelity (no momentum/variance state) — inference and fine-tuning only.
- **No normalization statistics.** Stored separately, see §2.

---

## 2. Required Artifacts Beyond the Checkpoint

A trained checkpoint alone is not enough to make a real-unit prediction. You also
need, all from `final_notebook/output-metrics/kagglephase1_output/`:

| File | Purpose |
|---|---|
| `normalization_stats.json` | `{mean, std}` per raw metric column — required to normalize new input and de-normalize model output back into real units (CPU-seconds, bytes) |
| `feature_cols.json` | Exact 27-column feature order the model expects, plus `target_idx` (which 4 of those 27 columns are the residual-anchor/prediction targets: indices `[0, 3, 4, 5]`) |
| `model_defs.py` (`kagglephase2_defs/`) | Source of `AdaptiveGRUModel` and every preprocessing/inference helper function |

---

## 3. Input Schema

```text
Raw input required per prediction:
  - A contiguous time series of the 7 raw metrics, per container, at 15-second
    intervals:
      container_cpu_usage_seconds_total, container_cpu_system_seconds_total,
      container_cpu_user_seconds_total, container_memory_usage_bytes,
      container_memory_working_set_bytes, container_memory_rss,
      container_memory_cache
  - Length: between 500 and 1000 most-recent timesteps (the adaptive window's
    valid range -- fewer than 500 cannot be scored; the model was never trained
    on shorter sequences)

Preprocessing required before the model sees anything (must exactly replicate
final_notebook/final-output/phase1 final.ipynb's Step 6):
  1. z-score normalize each of the 7 raw columns using normalization_stats.json's
     mean/std -- these are TRAIN-SPLIT statistics, frozen at training time, not
     recomputed from the new input
  2. per-container lag diffs (1, 2, 3 steps) + rolling mean/std (3-step window) on
     each of the 4 target columns, producing 27 total feature columns in the exact
     order given by feature_cols.json
  3. tensor shape: (batch=1, seq_len in [500,1000], 27), float32
```

**Format:** shape only — a `.csv`/DataFrame with 7 named columns, ordered
chronologically, is the natural real-world input; converting it to the 27-feature
tensor above is the caller's responsibility (there is no packaged
"predict_from_raw_dataframe()" convenience function anywhere in the project —
`Not found / Not verified` as an existing utility; a caller must replicate Phase 1's
Step 6 logic directly).

---

## 4. Output Schema

```text
Model output: (batch=1, 4) float32, in NORMALIZED (z-score) space
  Column order matches TARGET_NAMES: [cpu_usage, mem_usage, mem_working_set, mem_rss]

De-normalization (required before the output means anything):
  real_value = normalized_value * std + mean
  (per-column, using normalization_stats.json's stats for that specific
  target's raw column name)
```

**No anomaly score, no threshold, no classification output anywhere.** This model
produces continuous-valued forecasts only. If the intended downstream use is
anomaly/alerting, that decision layer does not exist in this project and would need
to be built on top of the forecast (e.g. comparing the forecast against a live reading
and flagging when the deviation is large — not implemented here).

---

## 5. Inference Process — Practical Example

```python
import json
import torch
from pathlib import Path

# 1. Load artifacts
ARTIFACT_DIR = Path("final_notebook/output-metrics/kagglephase1_output")
CKPT_DIR = Path("final_notebook/output-metrics/kagglephase3_output/checkpoints")

fc = json.load(open(ARTIFACT_DIR / "feature_cols.json"))
stats = json.load(open(ARTIFACT_DIR / "normalization_stats.json"))
target_idx = fc["target_idx"]                 # [0, 3, 4, 5]
target_names = fc["target_names"]              # ['cpu_usage', 'mem_usage', 'mem_working_set', 'mem_rss']
target_columns = fc["target_columns"]          # raw column names, same order
target_mean = [stats[c]["mean"] for c in target_columns]
target_std  = [stats[c]["std"]  for c in target_columns]

import sys
sys.path.insert(0, "final_notebook/output-metrics/kagglephase2_defs")
from model_defs import AdaptiveGRUModel

# 2. Rebuild the EXACT architecture used at training time -- residual_indices is
#    mandatory and is NOT recoverable from the checkpoint file itself (see §1)
HORIZON = 1   # 1, 2, or 3 -- selects which checkpoint / which forecast distance
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AdaptiveGRUModel(input_size=27, hidden_size=128, num_layers=2,
                         dropout=0.2, residual_indices=target_idx).to(device)
ckpt = torch.load(CKPT_DIR / f"gru_h{HORIZON}_static.pt", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# 3. Prepare input: X is a (1, L, 27) float32 tensor built per Section 3's
#    preprocessing steps, L in [500, 1000]; lengths tensor records the true length
#    for pack_padded_sequence
X = ...            # torch.Tensor, shape (1, L, 27)
lengths = torch.tensor([X.shape[1]])

# 4. Predict (normalized space)
with torch.no_grad():
    pred_norm = model(X.to(device), lengths)     # shape (1, 4)

# 5. De-normalize to real units
import numpy as np
pred_real = pred_norm.cpu().numpy()[0] * np.array(target_std) + np.array(target_mean)
result = dict(zip(target_names, pred_real.tolist()))
# result = {'cpu_usage': ..., 'mem_usage': ..., 'mem_working_set': ..., 'mem_rss': ...}
```

**No anomaly scoring or thresholding step exists** — the pipeline stops at `result`
above. This differs from the documentation brief's generic template
(`Anomaly Score -> Threshold -> Detection Result`), which does not apply to this
forecasting project; the practical flow that *does* apply is:

```text
Load model + artifacts
    |
Assemble 500-1000 step raw metric history for one container
    |
Normalize (train-split stats) + engineer 27 features
    |
Forward pass (packed variable-length sequence)
    |
De-normalize output (4 values, real units)
    |
Return forecast
```

---

## 6. External Integration Guide

### Required components checklist

- [ ] `gru_h{1,2,3}_static.pt` — pick whichever forecast horizon(s) the application needs
- [ ] `feature_cols.json`, `normalization_stats.json`
- [ ] `model_defs.py` (or at minimum the `AdaptiveGRUModel` class definition extracted
      from it)
- [ ] Python ≥3.8, PyTorch (version `Not pinned` — the training environment used
      `2.10.0+cu128`; a different build is not guaranteed to reproduce identical
      numeric output, and this project's own history includes a real
      CUDA-kernel-mismatch failure from an environment drift of exactly this kind —
      see §7)
- [ ] GPU optional for inference (a single forward pass on one window is cheap enough
      for CPU; **not benchmarked** in this project — inference latency is
      `Not found / Not verified`)

### Python application integration

Directly usable as shown in §5 — import `model_defs.py`, load a checkpoint,
call the model. This is the lowest-friction integration path since it requires no
new serving layer.

### REST API / FastAPI service (not implemented in this project — guidance only)

```python
from fastapi import FastAPI
import torch

app = FastAPI()
_models = {}   # horizon -> loaded AdaptiveGRUModel, loaded once at startup

@app.on_event("startup")
def load_models():
    for h in (1, 2, 3):
        # ... exact loading code from Section 5 ...
        _models[h] = model

@app.post("/predict/{horizon}")
def predict(horizon: int, history: list[dict]):
    # history: caller-supplied list of {timestamp, cpu_usage_seconds_total, ...}
    # dicts, >=500 rows -- caller is responsible for supplying raw metric history
    # in the schema of Section 3; this endpoint would run Section 3's preprocessing
    # + Section 5's forward pass + de-normalization and return `result`
    ...
```

This is a **design suggestion consistent with the actual model's interface**, not a
component that exists in the project. No `app.py`, `main.py`, or any web-framework
file exists anywhere in `module2/` — `Not found / Not verified` as an implemented
service.

### Containerized deployment

A Dockerfile would need: Python base image, `torch` matching the training CUDA build
(or CPU-only build if GPU inference isn't needed), the 3 `.pt` checkpoints, the 2
JSON artifact files, and `model_defs.py` (or the extracted class). **No Dockerfile
exists in this project currently** — this is guidance for a future integration step,
not a description of an existing artifact.

### Monitoring-pipeline integration

The natural fit, given the model's purpose (proactive scaling), is a sidecar or
periodic job that: (1) pulls the last 500–1000 samples of a container's metrics from
whatever monitoring system is in use (Prometheus, per the raw data's KPI naming
convention, is a plausible fit given the `container_*_seconds_total` counter-style
names), (2) runs the inference flow in §5, (3) hands the forecast to whatever scaling
decision logic exists downstream. **This integration is not implemented** — the
project ends at the forecast/evaluation stage.

---

## 7. Troubleshooting & Common Mistakes

These are drawn from **real failures encountered and fixed during this project's own
development** (not hypothetical):

| Symptom | Cause | Fix |
|---|---|---|
| Model loads, but predictions are silently wrong (no error) | Rebuilt `AdaptiveGRUModel` without passing `residual_indices` — the checkpoint's weights load fine into a structurally different (non-residual) model, producing garbage with zero error message | Always pass `residual_indices=target_idx` exactly as used at training; there is no runtime check that catches this mismatch |
| `AcceleratorError: CUDA error: no kernel image is available for execution on the device` | PyTorch build's compiled CUDA kernels don't match the runtime GPU's driver/architecture — encountered during this project's own Phase 2 development on a Kaggle session | Restart the session (often lands on a different GPU node); or switch the Kaggle Accelerator setting; or reinstall a PyTorch build matching `!nvidia-smi`'s reported CUDA version |
| `inspect.getsource()` raises `OSError: source code not available` when trying to export code from a notebook | Kaggle's kernel has no `__main__.__file__` set, which breaks Python's standard `inspect.getfile()` for classes defined directly in a notebook cell (functions are usually fine via linecache; classes are not, in this environment) | This project's actual fix (`phase2 final.ipynb` Step 7): embed source as literal JSON-escaped strings instead of using runtime introspection at all |
| Notebook `Run All` appears to halt silently, no error shown | Encountered in this project's Phase 2/3 development — Kaggle's batch/commit log streaming did not surface a real Python traceback that occurred underneath (turned out to be the CUDA error above) | Re-run interactively (not via commit/batch mode) to see the actual traceback; check the session's RAM/GPU indicator for a crashed state |
| Phase 3/4 can't find checkpoints/`model_defs.py` | The cross-notebook zip → Kaggle Dataset → Add Input handoff (§6 of `01_project_overview_and_architecture.md`) was skipped or pointed at the wrong dataset | Every final notebook's Step 1 cell raises an explicit `FileNotFoundError` naming exactly which dataset is missing — read that message, it names the required Dataset directly |
| `IndexError: shape mismatch` when computing target values from a windows/feature array | NumPy paired advanced indexing (`arr[row_indices, col_indices]`) was used where `arr[row_indices][:, col_indices]` (two separate indexing steps) was needed — a real bug encountered and fixed in this project's `phase3 final.ipynb` Step 4 | Use `arr[rows][:, cols]` (or `arr[rows[:, None], cols]`) whenever selecting many rows AND several specific columns together, never `arr[rows, cols]` unless the two index arrays are meant to pair up elementwise |

---

## 8. Known Limitations (for integration planning)

1. **No pinned dependency versions** — reproducing exact numeric output on a different
   PyTorch/CUDA build is not guaranteed.
2. **No batching support demonstrated for multi-container simultaneous inference** in
   any example in this project — the training/eval code processes windows in batches
   internally, but no documented API exists for "score N containers at once" as an
   external caller.
3. **`AdaptiveThreshold`'s confidence band is computed but not exposed** — an
   integrator wanting a confidence interval alongside the point forecast would need to
   extract this from `model_defs.py`'s `AdaptiveThreshold` class directly; it is not
   wired into any prediction function.
4. **The adaptive window's length rule is historical, not live** (§7 of
   `02_data_pipeline_and_methodology.md`) — a production integration feeding genuinely
   new data would need to decide independently what window length to supply, since
   the training-time rule was never designed to be queried at inference time for new,
   unseen containers.
5. **Only `complex_case1`-style single-deployment data was ever used for
   training/testing** — no evidence exists in this project of the model's behavior on
   a genuinely different deployment topology at inference time (the drift-detection
   Phase 4 stream is still evaluated within the same case, just injected with
   synthetic bursts, not moved to a different case).
