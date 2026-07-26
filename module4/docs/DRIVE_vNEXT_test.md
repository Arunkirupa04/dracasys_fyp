# Drive layout — `vNEXT_test` (FUSE-safe)

**Last updated:** 2026-07-11  
**Helpers:** readable code cells inside the three notebooks (same source as `notebook/mdc_*.py`; no base64 blobs)  
**Notebooks:** `mdc_preprocess_vNext_mc.ipynb`, `mdc_model_vNext.ipynb`, `mdc_drift_aware.ipynb`

Repo copies of `mdc_drive_io.py` / `mdc_eval_utils.py` / `mdc_label_map.py` are for local tests only.

---

## Canonical folder (all three notebooks)

```
My Drive /
  vNEXT_test /
    processed /          ← preprocess outputs
    runs /               ← model checkpoint, scores, metrics, §19 plots
    drift_aware /        ← drift-aware outputs + thesis figures
    SYNC_RECEIPT.json    ← visible sync marker after every push
```

**Colab local staging (always write here first):**

```
/content/vNEXT_test_local /
  processed /
  runs /
  drift_aware /
```

---

## Why “I pushed but Drive UI is empty”

Colab’s Drive mount is **FUSE**. Direct writes to `/content/drive/MyDrive/...` often:

1. Look successful in the notebook (`Copied -> ...`)
2. Never appear (or appear late / never) in [drive.google.com](https://drive.google.com)

### Past failures in this project

| Failure | Cause | Fix now |
|---------|-------|---------|
| `Mountpoint must not already contain files` | `force_remount=True` / remount on live mount | Mount only if `MyDrive` missing |
| UI empty after “Copied” | Write-only-to-FUSE | Local stage → verified `push_file` |
| Wrong Google account | Colab ≠ Drive browser account | Same account; check `SYNC_RECEIPT.json` |
| File on shared folder ID only | Not under My Drive tree | Always push under `vNEXT_test` |

---

## Protocol (every notebook)

1. **Save locally** under `/content/vNEXT_test_local/...`
2. **Push cell** uses `mdc_drive_io.push_file` / `push_tree`
3. **Verify** source size == destination size
4. **Write** `vNEXT_test/SYNC_RECEIPT.json`
5. **Check UI:** My Drive → **vNEXT_test** → subfolder (+ receipt at root)

If UI still empty after 2 minutes: hard-refresh Drive, confirm account, re-run push cell only.

---

## What each notebook pushes / pulls

| Notebook | Pulls from | Pushes to |
|----------|------------|-----------|
| `mdc_preprocess_vNext_mc` | (raw data) | `vNEXT_test/processed/` |
| `mdc_model_vNext` | `processed/` | `vNEXT_test/runs/` |
| `mdc_drift_aware` | `processed/` + `runs/` | `vNEXT_test/drift_aware/` |

Legacy paths (`Module4_MDC/...`) remain as **read fallbacks only**, not write targets.

---

## Colab upload checklist

Upload **only the three notebooks** — helpers are normal readable code cells (run the Drive/eval helper cell once):

- `mdc_preprocess_vNext_mc.ipynb`
- `mdc_model_vNext.ipynb`
- `mdc_drift_aware.ipynb`

No separate `.py` upload required for Colab. Repo `.py` files remain the source of truth for local tests / diffs.

---

## Quick verify snippet (Colab)

```python
from pathlib import Path
root = Path('/content/drive/MyDrive/vNEXT_test')
print('exists', root.is_dir())
print('children', sorted(p.name for p in root.iterdir()) if root.is_dir() else None)
print('runs', list((root/'runs').glob('*'))[:10] if (root/'runs').is_dir() else None)
```
