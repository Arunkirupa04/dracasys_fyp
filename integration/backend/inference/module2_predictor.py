"""
Module 2 — Short-Term Resource Prediction (AdaptiveGRUModel).

Loads production_model.pt (self-contained bundle), builds one GRU instance
per forecast horizon (1=15s, 2=30s, 3=45s ahead), and produces de-normalized
4-value resource forecasts from a (1, 1000, 27) input tensor.

Preprocessing is done here; the caller passes a raw (N, 7) numpy array where
N >= 1000, with columns in the order defined in config and normalization_stats.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from backend.config import MODEL_PATHS

RAW_COLS = [
    "container_cpu_usage_seconds_total",
    "container_cpu_system_seconds_total",
    "container_cpu_user_seconds_total",
    "container_memory_usage_bytes",
    "container_memory_working_set_bytes",
    "container_memory_rss",
    "container_memory_cache",
]

TARGET_COLS = [
    "container_cpu_usage_seconds_total",
    "container_memory_usage_bytes",
    "container_memory_working_set_bytes",
    "container_memory_rss",
]

TARGET_NAMES = ["cpu_usage", "mem_usage", "mem_working_set", "mem_rss"]


class Module2Predictor:
    name = "Module 2 — Short-Term Resource Prediction"

    def __init__(self):
        bundle_path = MODEL_PATHS["m2_bundle"]
        defs_path   = MODEL_PATHS["m2_defs"]
        norm_path   = MODEL_PATHS["m2_norm"]

        # Make model_defs.py importable
        sys.path.insert(0, str(defs_path.parent))
        from model_defs import AdaptiveGRUModel  # noqa: PLC0415

        self.bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        arch = self.bundle["architecture"]

        with open(norm_path) as f:
            self._norm_stats = json.load(f)

        tgt = self.bundle["targets"]
        self._target_mean = np.array(tgt["target_mean"], dtype=np.float64)
        self._target_std  = np.array(tgt["target_std"],  dtype=np.float64)
        self._feature_cols = self.bundle["feature_cols"]   # ordered list of 27 names

        # Build one model per horizon and load weights
        self._models: dict[int, torch.nn.Module] = {}
        for h in (1, 2, 3):
            m = AdaptiveGRUModel(
                input_size=arch["input_size"],
                hidden_size=arch["hidden_size"],
                num_layers=arch["num_layers"],
                dropout=arch["dropout"],
                residual_indices=arch["residual_indices"],
            )
            m.load_state_dict(self.bundle["horizons"][h]["model_state_dict"])
            m.eval()
            self._models[h] = m

        print(f"[M2] Loaded AdaptiveGRUModel — horizons H1/H2/H3, "
              f"input_size={arch['input_size']}, hidden={arch['hidden_size']}")

    # ------------------------------------------------------------------
    def _preprocess(self, raw_np: np.ndarray) -> torch.Tensor:
        """
        raw_np : (N, 7) float64, columns in RAW_COLS order, N >= 1000.
        Returns (1, 1000, 27) float32 tensor ready for model forward pass.
        """
        assert raw_np.shape[1] == 7, f"Expected 7 columns, got {raw_np.shape[1]}"
        assert raw_np.shape[0] >= 1000, f"Need >= 1000 rows, got {raw_np.shape[0]}"

        # Use last 1000 rows
        arr = raw_np[-1000:].copy().astype(np.float64)

        # Step 1: z-score normalize all 7 raw columns
        col_arr = {}
        for j, col in enumerate(RAW_COLS):
            mean = self._norm_stats[col]["mean"]
            std  = self._norm_stats[col]["std"]
            norm = (arr[:, j] - mean) / std
            col_arr[col] = norm

        # Step 2: compute 20 engineered features (lag diffs + rolling) on 4 target cols
        # Names must match bundle feature_cols exactly: _DIFF_1, _ROLLING_MEAN_3, etc.
        eng = {}
        for col in TARGET_COLS:
            s = col_arr[col]
            eng[f"{col}_DIFF_1"] = np.concatenate([[0],       s[1:] - s[:-1]])
            eng[f"{col}_DIFF_2"] = np.concatenate([[0, 0],    s[2:] - s[:-2]])
            eng[f"{col}_DIFF_3"] = np.concatenate([[0, 0, 0], s[3:] - s[:-3]])
            roll_mean = np.full(1000, np.nan)
            roll_std  = np.full(1000, np.nan)
            for idx in range(2, 1000):
                window = s[idx - 2: idx + 1]
                roll_mean[idx] = window.mean()
                roll_std[idx]  = window.std(ddof=0)
            roll_mean[:2] = roll_mean[2]
            roll_std[:2]  = 0.0
            eng[f"{col}_ROLLING_MEAN_3"] = roll_mean
            eng[f"{col}_ROLLING_STD_3"]  = roll_std

        # Step 3: assemble 27 features in bundle's declared order
        all_feats = {**col_arr, **eng}
        matrix = np.stack([all_feats[fc] for fc in self._feature_cols], axis=1)  # (1000, 27)
        matrix = np.nan_to_num(matrix, nan=0.0).astype(np.float32)

        return torch.from_numpy(matrix).unsqueeze(0)  # (1, 1000, 27)

    # ------------------------------------------------------------------
    def predict(self, raw_np: np.ndarray) -> dict:
        """
        raw_np : (N, 7) float64 container metrics, N >= 1000.
        Returns dict with forecasts for all 3 horizons.
        """
        X = self._preprocess(raw_np)
        lengths = torch.tensor([X.shape[1]])

        horizons_out = {}
        horizon_seconds = {1: 15, 2: 30, 3: 45}

        for h, model in self._models.items():
            with torch.no_grad():
                pred_norm = model(X, lengths).numpy()[0]          # (4,)
            pred_real = pred_norm * self._target_std + self._target_mean  # de-normalize

            horizons_out[f"h{h}"] = {
                "horizon_seconds": horizon_seconds[h],
                "cpu_usage":       round(float(pred_real[0]), 2),
                "mem_usage_mb":    round(float(pred_real[1]) / 1e6, 2),
                "mem_wss_mb":      round(float(pred_real[2]) / 1e6, 2),
                "mem_rss_mb":      round(float(pred_real[3]) / 1e6, 2),
            }

        return {
            "module":   "m2",
            "status":   "ok",
            "label":    "Short-Term Resource Prediction",
            "horizons": horizons_out,
        }
