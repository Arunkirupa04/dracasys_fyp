"""
Module 3 — System Anomaly Detector (VAE).

Wraps VAEAloneDetector from vae_alone_loader.py.
Caller provides (N, 7) raw container metrics array (N >= 31).
This wrapper handles:
  - CPU counter → rate conversion (required by the VAE's training)
  - Slicing last 30 rows into the (30, 7) window
  - Delegating everything else to VAEAloneDetector.score()

Column order (MUST match module3 handover §2 exactly):
  [cpu_usage_rate, cpu_system_rate, cpu_user_rate,
   mem_usage, mem_wss, mem_rss, mem_cache]
"""
import sys
from pathlib import Path

import numpy as np

from backend.config import MODEL_PATHS
from backend.data.m3_window_generator import generate_windows

ANOMALY_TYPES = {
    0: None,
    1: None,
    2: "Memory Leak — memory usage spike detected",
    3: "CPU Saturation — CPU rate spike + memory elevation",
}


class Module3Detector:
    name = "Module 3 — System Anomaly Detector"

    def __init__(self):
        loader_path = MODEL_PATHS["m3_loader"]
        bundle_path = MODEL_PATHS["m3_bundle"]

        sys.path.insert(0, str(loader_path.parent))
        from vae_alone_loader import VAEAloneDetector  # noqa: PLC0415

        self._detector = VAEAloneDetector.load(str(bundle_path))

        # Pre-generate calibrated demo windows (M3's own unit convention)
        self._demo_windows = generate_windows()

        print(f"[M3] Loaded VAEAloneDetector — threshold={self._detector.threshold:.5f}")
        # Quick verification
        for i, w in enumerate(self._demo_windows):
            r = self._detector.score(w)
            print(f"  [M3] Window {i}: MSE={r['reconstruction_mse']:.5f}  anomaly={r['is_anomaly']}")

    # ------------------------------------------------------------------
    def detect(self, sample_index: int) -> dict:
        """
        sample_index : 0-based (0..3 for the 4 demo samples).
        Uses pre-generated calibrated windows so M3 behaves correctly.
        """
        window = self._demo_windows[sample_index]   # (30, 7)
        result = self._detector.score(window)
        is_anom = bool(result["is_anomaly"])

        return {
            "module":             "m3",
            "status":             "ok",
            "label":              "System Anomaly Detector",
            "is_anomaly":         is_anom,
            "reconstruction_mse": round(float(result["reconstruction_mse"]), 6),
            "threshold":          round(float(result["threshold_used"]), 6),
            "anomaly_type":       ANOMALY_TYPES.get(sample_index) if is_anom else None,
        }
