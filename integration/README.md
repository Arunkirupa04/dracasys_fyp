# DracaSys — Integrated Demo

4-module AI demo: long-term forecasting (M1) + short-term resource prediction (M2) + system anomaly detection (M3) + security anomaly detection (M4), streamed live to a React UI over SSE.

## ⚡ Quick Reference — Run All 3 Services

Three terminals, run in this order. Copy-paste blocks below (Windows/PowerShell paths shown; adjust `cd` for other shells).

| # | Service | Command | Port | Wait for |
|---|---|---|---|---|
| 1 | **Module 1** (Prophet+GRU) | see below — **uses its own short-path venv** | `:8000` | `Uvicorn running on http://0.0.0.0:8000` |
| 2 | **Integration backend** | `cd integration && python run.py` | `:5000` | `All models loaded. Ready.` |
| 3 | **Frontend** | `cd integration/frontend && npm run dev` | `:5173` | `VITE ... ready` |

```powershell
# Terminal 1 — Module 1 (only needed once: create the venv + install deps)
python -m venv C:\v\m1env
C:\v\m1env\Scripts\python.exe -m pip install -r module1\requirements.txt
# Then every time:
C:\v\m1env\Scripts\python.exe module1\run.py

# Terminal 2 — Integration backend (from repo root)
cd integration
python run.py

# Terminal 3 — Frontend (from repo root)
cd integration\frontend
npm run dev
```

Then open **http://localhost:5173**, click **▶ Start Stream**.

> **Why a separate venv for Module 1?** Its `tensorflow` dependency fails to install in a deeply-nested path on Windows (`OSError` extracting `tensorflow/python/...` — hits `MAX_PATH` because Windows long-path support is disabled on this machine: `LongPathsEnabled=0`). Installing into a short path (`C:\v\m1env`) sidesteps this without needing admin rights. If you'd rather fix it system-wide: enable long paths via `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled=1` (needs admin + reboot), then you can use a normal venv anywhere and `pip install -r module1/requirements.txt` directly.

**If Module 1 isn't running, the demo still works** — its card shows a clear "Module 1 unreachable" message instead of blocking M2/M3/M4, which stream normally.

**Verify each service:**
```
curl http://localhost:8000/health   → {"status":"healthy","model_loaded":true,...}
curl http://localhost:5000/api/status → {"ready":true,"modules":{...},"demo_samples":10}
```

**To stop:** Ctrl+C in each terminal. **To restart the demo:** click Reset in the UI, or just re-click Start Stream after a full page refresh.

## Ports

| Service | Port | Notes |
|---|---|---|
| Module 1 (Prophet + GRU) | **8000** | Separate FastAPI microservice, own repo folder (`module1/`), TensorFlow stack, own venv (`C:\v\m1env`) |
| Integration backend | **5000** | This app — FastAPI, loads M2/M3/M4 in-process, calls M1 over HTTP |
| Frontend (Vite dev server) | **5173** | Proxies `/api/*` → `localhost:5000` |

Module 1 keeps its own default port (8000) unmodified — the integration backend runs on 5000 instead of the more common 8000 to avoid the clash.

## Detailed Setup (first-time only)

### 1. Module 1 service

```powershell
python -m venv C:\v\m1env
C:\v\m1env\Scripts\python.exe -m pip install -r module1\requirements.txt
C:\v\m1env\Scripts\python.exe module1\run.py       # listens on :8000
```

Verify: `curl http://localhost:8000/health` → `{"status":"healthy","model_loaded":true,...}`

If this service isn't running, the demo still works — the M1 card shows a clear "Module 1 unreachable" message instead of blocking M2/M3/M4.

### 2. Integration backend

```bash
cd integration
pip install -r requirements.txt   # Python 3.10+ required
python run.py                     # listens on :5000
```

Wait for "All models loaded. Ready." in the terminal, then verify:
```
GET http://localhost:5000/api/status
```

### 3. Frontend

```bash
cd integration/frontend
npm install
npm run dev
```

Open **http://localhost:5173** in your browser.

### 4. Run the demo

1. Click **▶ Start Stream** in the browser.
2. Module 1's 24h forecast request fires immediately in the background (~5-30s, Prophet refit) — its card shows a spinner until the result arrives, independent of the sample ticks.
3. Watch 10 samples stream through M2/M3/M4 at 4-second intervals (~40s total).
4. Demo narrative across the 10 samples (see `backend/data/m3_window_generator.py` and `backend/inference/module4_detector.py`):
   - Sample 4 — M3 flags an early memory-leak warning
   - Sample 7 — M3 flags a CPU-saturation spike
   - Sample 8 — M4 confirms the one real, model-verified attack in the held-out test set (CVE-2020-13379 Grafana SSRF)
   - Sample 10 — M3 flags a severe combined system degradation (aftermath)
5. Every module card now includes a chart: M1/M2 show actual → predicted trend lines, M3/M4 show a score/threshold trend across all completed samples.
6. Click **↺ Reset** to run again (re-fires the M1 forecast too).

---

## Architecture

```
integration/
├── run.py                    Entry point — uvicorn on INTEGRATION_HOST:INTEGRATION_PORT
├── models/
│   ├── module2/   production_model.pt + model_defs.py + normalization_stats.json
│   ├── module3/   vae_cc1_final_model.pt + vae_alone_loader.py
│   └── module4/   model_hpo_best_vnext.pt + sequence_bottleneck_ae.py
├── backend/
│   ├── main.py               FastAPI app (startup + /api/status + /api/stream)
│   ├── config.py             All paths, ports, and demo params
│   ├── inference/            One wrapper per module (module1_client is HTTP, rest in-process)
│   ├── data/                 Synthetic container stream + M1 demo request JSON
│   └── streaming/            SSE event loop
└── frontend/                 React + Vite dark-theme UI
```

## Module 1 integration approach

Module 1 is a fully separate FastAPI microservice (TensorFlow + Prophet), never imported in-process by this backend. `backend/inference/module1_client.py` calls it asynchronously over HTTP, following the documented contract in `module1/docs/api/`:

- Fired once per demo run via `asyncio.create_task()` at stream start — it never blocks the M2/M3/M4 5-second tick loop.
- Uses a real 672-step (7-day) holdout history for an unseen container (`module1/examples/unseen_c_10312_forecast_request.json`, copied to `backend/data/m1_request.json`) — satisfies the API's recommended history length and is never derived from the M2/M3 synthetic stream (different units/interval).
- Result is emitted via SSE as soon as it resolves (checked after every sample), or awaited with a 90s bound if the 4-sample loop finishes first.
- If the service is unreachable or times out, the UI shows a clear error state instead of blocking the rest of the demo.

## Data Sources

| Stream | Modules | Data |
|---|---|---|
| Container metrics (Stream A) | M2 | Synthetic (1010 rows = 1000 history + 10 demo, 7 features), with a gentle load ramp over the demo tail so the actual-vs-predicted chart shows a real trend. |
| System windows (Stream B) | M3 | Its own calibrated synthetic windows (`backend/data/m3_window_generator.py`) matched to its trained scaler — 3 of the 10 windows are anomalous (mem leak, CPU saturation, combined degradation). |
| Network flow windows (Stream C) | M4 | Real test windows from `windows_vnext.npz` (Strategy A) — 9 model-confirmed benign + the 1 model-confirmed attack in the whole 5153-window test set. |
| 7-day CPU history (Stream D) | M1 | Real unseen-container holdout (`module1/examples/unseen_c_10312_forecast_request.json`), 672 × 15-min steps. |

## Module Outputs

| Module | Type | Output |
|---|---|---|
| M1 | Forecast | 96-step (24h) `predicted_cpu_percent` + Prophet/GRU components + 12h actual history tail, fetched over HTTP |
| M2 | Forecast | actual current cpu/mem/wss/rss + forecasts at 15s, 30s, 45s ahead |
| M3 | Anomaly | reconstruction_mse, is_anomaly, threshold, anomaly_type |
| M4 | Anomaly | score, is_anomaly, threshold, attack_type |

## Charts

Every module card renders an inline SVG chart (no charting library — plain `<svg>`, scales via `viewBox`):

- **M1 / M2** — `ActualPredictedChart`: a solid "actual" line (recent history / streamed readings) continuing into a dashed "predicted" line (forecast).
- **M3 / M4** — `ScoreTrendChart`: reconstruction MSE (log scale, M3) or anomaly score (linear scale, M4) across all completed samples, with a dashed threshold line and red/green points per sample.

See `frontend/src/components/charts.jsx`.
