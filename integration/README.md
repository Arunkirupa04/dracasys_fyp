# DracaSys — Integrated Demo

4-module AI demo: short-term resource prediction (M2) + system anomaly detection (M3) + security anomaly detection (M4) + Module 1 placeholder.

## Quick Start

### 1. Backend

```bash
cd integration

# Install Python deps (Python 3.10+ required)
pip install -r requirements.txt

# Start the FastAPI server
uvicorn backend.main:app --reload --port 8000
```

Wait for "All models loaded. Ready." in the terminal, then verify:
```
GET http://localhost:8000/api/status
```

### 2. Frontend

```bash
cd integration/frontend

npm install
npm run dev
```

Open **http://localhost:5173** in your browser.

### 3. Run the demo

1. Click **▶ Start Stream** in the browser.
2. Watch 4 samples stream through all 4 modules at 5-second intervals.
3. Samples 3 & 4 trigger anomaly detections in M3 (system) and M4 (security).
4. Click **↺ Reset** to run again.

---

## Architecture

```
integration/
├── models/
│   ├── module2/   production_model.pt + model_defs.py + normalization_stats.json
│   ├── module3/   vae_cc1_final_model.pt + vae_alone_loader.py
│   └── module4/   model_hpo_best_vnext.pt + sequence_bottleneck_ae.py
├── backend/
│   ├── main.py               FastAPI app (startup + /api/status + /api/stream)
│   ├── config.py             All paths and demo params
│   ├── inference/            One wrapper per module
│   ├── data/                 Synthetic container stream generator
│   └── streaming/            SSE event loop
└── frontend/                 React + Vite dark-theme UI
```

## Data Sources

| Stream | Modules | Data |
|---|---|---|
| Container metrics (Stream A) | M2 + M3 | Synthetic (1004 rows, 7 features). Rows 1003-1004 injected with memory leak + CPU spike. |
| Network flow windows (Stream B) | M4 | Real test windows from `windows_vnext.npz` (Strategy A). Windows 3-4 are real labeled attacks from the MDC dataset. |

## Module Outputs

| Module | Type | Output |
|---|---|---|
| M1 | Stub | "not_implemented" |
| M2 | Forecast | cpu/mem/wss/rss at 15s, 30s, 45s ahead |
| M3 | Anomaly | reconstruction_mse, is_anomaly, threshold |
| M4 | Anomaly | score, is_anomaly, threshold, attack_type |
