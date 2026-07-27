from pathlib import Path

# Absolute path to integration/
INTEGRATION_ROOT = Path(__file__).resolve().parent.parent

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

DEMO_DELAY_SECONDS = 5.0
N_DEMO_SAMPLES = 4
N_HISTORY_ROWS = 1000   # rows pre-seeded before the 4 demo samples
M2_WINDOW = 1000        # rows fed to GRU (must be in [500, 1000])
M3_WINDOW = 30          # rows fed to VAE
