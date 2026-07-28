import os
from pathlib import Path

from backend.config_loader import demo_config, load_config

# Absolute path to integration/
INTEGRATION_ROOT = Path(__file__).resolve().parent.parent

_cfg = load_config()
_demo = demo_config()

MODELS_DIR = INTEGRATION_ROOT / "models"

MODEL_PATHS = {
    "m2_bundle":   MODELS_DIR / "module2" / "production_model.pt",
    "m2_defs":     MODELS_DIR / "module2" / "model_defs.py",
    "m2_norm":     MODELS_DIR / "module2" / "normalization_stats.json",
    "m3_bundle":   MODELS_DIR / "module3" / "vae_cc1_final_model.pt",
    "m3_loader":   MODELS_DIR / "module3" / "vae_alone_loader.py",
    "m4_bundle":   MODELS_DIR / "module4" / "model_hpo_best_vnext.pt",
    "m4_ae_class": MODELS_DIR / "module4" / "sequence_bottleneck_ae.py",
    "m4_windows":  INTEGRATION_ROOT / "backend" / "data" / "demo_windows_m4.npy",
    "m4_npz":      INTEGRATION_ROOT.parent / "module4" / "notebook" / "final" /
                   "output-metrics" / "windows_vnext_processed" / "windows_vnext.npz",
    "m4_labels":   INTEGRATION_ROOT.parent / "module4" / "notebook" / "final" /
                   "output-metrics" / "windows_vnext_processed" / "mdc_label_map.json",
}

DEMO_DELAY_SECONDS = float(_demo.get("demo_delay_seconds", 4.0))
N_DEMO_SAMPLES = int(_demo.get("n_demo_samples", 10))
N_HISTORY_ROWS = int(_demo.get("n_history_rows", 1000))
M2_WINDOW = int(_demo.get("m2_window", 1000))
M3_WINDOW = int(_demo.get("m3_window", 30))

# -----------------------------------------------------------------------------
# Module 1 — Long-Term Forecasting (Prophet + GRU), served as a separate
# FastAPI microservice (module1/run.py). Never imported in-process — this
# stack uses TensorFlow + Prophet which is intentionally kept isolated from
# the PyTorch-only integration backend. Override via env vars if needed.
# -----------------------------------------------------------------------------
M1_BASE_URL         = os.environ.get(
    "M1_BASE_URL",
    _cfg.get("services", {}).get("module1_base_url", "http://localhost:8000"),
)
M1_REQUEST_PATH     = INTEGRATION_ROOT / "backend" / "data" / "m1_request.json"
M1_TIMEOUT_SECONDS  = float(os.environ.get("M1_TIMEOUT_SECONDS", "120"))
M1_HEALTH_TIMEOUT_SECONDS = 5.0

# -----------------------------------------------------------------------------
# Integration backend server (this FastAPI app). Module 1 keeps its own
# default port 8000, so this app runs on 5000 to avoid a conflict.
# -----------------------------------------------------------------------------
INTEGRATION_HOST = os.environ.get("INTEGRATION_HOST", "0.0.0.0")
INTEGRATION_PORT = int(os.environ.get(
    "INTEGRATION_PORT",
    str(_cfg.get("services", {}).get("integration_port", 5000)),
))
