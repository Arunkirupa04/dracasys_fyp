"""
Module 4 — Security Anomaly Detector (SequenceBottleneckAE).

Loads model_hpo_best_vnext.pt and scores pre-made (B, 10, 163) windows.
For the demo, windows are taken from the actual test set (windows_vnext.npz)
using Strategy A: real benign + real attack windows, with ground-truth attack
type labels from y_test_multiclass + mdc_label_map.json.

Scoring protocol (from MODEL_HANDOVER.md §5):
  err = (recon - x)^2 / feat_std  (feat_std clamped at 1e-6)
  raw = 0.3 * mean(err) + 0.7 * max_over_time(mean_over_features(err))
  score = -raw   (invert=True)
  alert = score >= threshold
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from backend.config import MODEL_PATHS


class Module4Detector:
    name = "Module 4 — Security Anomaly Detector"

    LABEL_DESCRIPTIONS = {
        0:  "Benign",
        1:  "CVE-2020-13379 (Grafana SSRF)",
        2:  "Node-RED Reconnaissance",
        3:  "Node-RED RCE",
        4:  "Node-RED Container Escape",
        5:  "CVE-2021-43798",
        6:  "CVE-2019-20933",
        7:  "CVE-2021-30465",
        8:  "CVE-2021-25741 (K8s Volume Escape)",
        9:  "CVE-2022-23648",
        10: "CVE-2019-5736 (runc Escape)",
        11: "DSB Nuclei Scan",
    }

    def __init__(self):
        bundle_path  = MODEL_PATHS["m4_bundle"]
        ae_path      = MODEL_PATHS["m4_ae_class"]
        npz_path     = MODEL_PATHS["m4_npz"]

        # Import SequenceBottleneckAE
        sys.path.insert(0, str(ae_path.parent))
        from sequence_bottleneck_ae import SequenceBottleneckAE  # noqa: PLC0415

        ckpt = torch.load(bundle_path, map_location="cpu", weights_only=False)
        assert ckpt["format"] == "mdc_vnext_hpo_standalone"

        arch          = ckpt["arch"]
        self._feat_std = torch.as_tensor(ckpt["feat_std"], dtype=torch.float32)
        proto         = ckpt["score_protocol"]
        self._mean_w  = float(proto["mean_w"])
        self._max_w   = float(proto["max_w"])
        self._invert  = bool(proto["invert"])
        self._threshold = float(ckpt["metrics_hpo"]["threshold"])
        self._T       = int(ckpt["T"])

        model = SequenceBottleneckAE(
            n_features    = int(ckpt["n_features"]),
            d_model       = arch["d_model"],
            nhead         = arch["nhead"],
            num_enc_layers= arch["num_enc_layers"],
            num_dec_layers= arch["num_dec_layers"],
            dim_ff        = arch["dim_ff"],
            dropout       = arch["dropout"],
            bottleneck_dim= arch["bottleneck_dim"],
            max_len       = max(self._T + 10, 200),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        self._model = model

        # Load demo windows from real test set (Strategy A)
        self._demo_windows, self._demo_labels, self._demo_multiclass = \
            self._load_demo_windows(npz_path)

        print(
            f"[M4] Loaded SequenceBottleneckAE HPO_best — "
            f"threshold={self._threshold:.4f}, "
            f"demo windows: {self._demo_windows.shape}"
        )

    # ------------------------------------------------------------------
    def _load_demo_windows(self, npz_path: Path):
        """
        Pick 4 demo windows from windows_vnext.npz, all pre-verified against
        the model's actual scores (not just ground truth labels):
          index 0, 1 → windows the model scores as benign  (score < threshold)
          index 2    → CVE/exploit window the model CONFIRMS as alert
          index 3    → Recon window the model CONFIRMS as alert
        """
        z = np.load(str(npz_path))
        X_test = z["X_test"].astype(np.float32)          # (5153, 10, 163)
        y_test = z["y_test"].astype(np.int32)             # binary 0/1
        y_mc   = z["y_test_multiclass"].astype(np.int32)  # multiclass

        # Score all test windows in batches to find verified true positives
        print("[M4] Pre-scoring test windows to find verified alerts…")
        BATCH = 256
        all_scores = []
        for start in range(0, len(X_test), BATCH):
            batch = X_test[start: start + BATCH]
            s = self._score(batch)
            all_scores.append(s)
        all_scores = np.concatenate(all_scores)   # (5153,)

        model_alerts = all_scores >= self._threshold   # model-confirmed alerts

        # Verified benign: ground truth 0 AND model says benign
        verified_benign = np.where((y_test == 0) & (~model_alerts))[0]
        # Verified exploits: ground truth 1, exploit label, model confirms
        exploit_labels = np.isin(y_mc, [1, 3, 8])
        verified_exploit = np.where((y_test == 1) & exploit_labels & model_alerts)[0]
        # Verified recon: ground truth 1, recon label, model confirms
        recon_labels = np.isin(y_mc, [2, 11])
        verified_recon = np.where((y_test == 1) & recon_labels & model_alerts)[0]

        # Fallback: any model-confirmed attack if specific types not available
        any_verified_attack = np.where((y_test == 1) & model_alerts)[0]
        if len(verified_exploit) == 0:
            verified_exploit = any_verified_attack
        if len(verified_recon) == 0:
            verified_recon = any_verified_attack

        print(f"  verified benign: {len(verified_benign)}, "
              f"exploit TPs: {len(verified_exploit)}, "
              f"recon TPs: {len(verified_recon)}")

        chosen = [
            verified_benign[5],    # sample 1 — model-confirmed benign
            verified_benign[20],   # sample 2 — model-confirmed benign
            verified_exploit[0],   # sample 3 — model-confirmed exploit
            verified_recon[0],     # sample 4 — model-confirmed recon
        ]

        windows = np.stack([X_test[i] for i in chosen], axis=0)   # (4, 10, 163)
        labels  = np.array([y_test[i] for i in chosen])
        mc      = np.array([y_mc[i]   for i in chosen])
        return windows, labels, mc

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _score(self, x_np: np.ndarray) -> np.ndarray:
        """x_np: (B, 10, 163) float32 → scores (B,)"""
        x = torch.from_numpy(np.asarray(x_np, dtype=np.float32))
        recon = self._model(x)
        err = (recon - x).pow(2)
        fs = self._feat_std.clamp_min(1e-6).view(1, 1, -1)
        err = err / fs
        s = self._mean_w * err.mean(dim=(1, 2)) + \
            self._max_w  * err.mean(dim=2).amax(dim=1)
        if self._invert:
            s = -s
        return s.cpu().numpy()

    # ------------------------------------------------------------------
    def detect(self, sample_index: int) -> dict:
        """
        sample_index : 0-based (0..3 for 4 demo samples)
        Returns anomaly result dict.
        """
        window = self._demo_windows[sample_index : sample_index + 1]  # (1, 10, 163)
        scores = self._score(window)
        score  = float(scores[0])
        is_anom = score >= self._threshold
        mc_label = int(self._demo_multiclass[sample_index])
        attack_name = self.LABEL_DESCRIPTIONS.get(mc_label, f"Label {mc_label}")

        return {
            "module":      "m4",
            "status":      "ok",
            "label":       "Security Anomaly Detector",
            "is_anomaly":  bool(is_anom),
            "score":       round(score, 6),
            "threshold":   round(self._threshold, 6),
            "attack_type": attack_name if is_anom else None,
            "ground_truth_label": mc_label,
        }
