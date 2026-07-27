"""
VAE-Alone Anomaly Detector — Module 3 (standalone, for external integration)

Self-contained loader for `vae_cc1_final_model.pt`. This is the base VAE model
only — no adaptive thresholding, no drift detection/incremental learning (see
the separate drift-aware package if you need those). Everything required for
inference (weights, PCA, scaler, static threshold) is inside the one .pt file.

Reported performance at the bundled threshold (val_p99 = 0.098):
  cc1_test (in-distribution):  PR-AUC=0.601, F1=0.618, Precision=0.630, Recall=0.605
  drift_cc2 (drift eval set):  PR-AUC=0.409, F1=0.276, Precision=0.168, Recall=0.772

Usage:
    from vae_alone_loader import VAEAloneDetector
    detector = VAEAloneDetector.load("vae_cc1_final_model.pt")
    result = detector.score(raw_window_30x7)   # raw_window_30x7: np.ndarray shape (30, 7)
"""

import numpy as np
import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden2, latent_dim)
        self.fc_lv = nn.Linear(hidden2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden2), nn.ReLU(),
            nn.Linear(hidden2, hidden1), nn.ReLU(),
            nn.Linear(hidden1, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), torch.clamp(self.fc_lv(h), -10, 10)

    def decode(self, z):
        return self.decoder(z)

    @torch.no_grad()
    def anomaly_score(self, x):
        self.eval()
        mu, _ = self.encode(x)
        return ((self.decode(mu) - x) ** 2).mean(dim=1)


class VAEAloneDetector:
    """Stateless — every call is independent, no per-container memory. Safe to
    share a single instance across threads/requests for inference-only use."""

    # Required raw-feature order and meaning — see README for full detail.
    FEATURE_COLS = [
        "container_cpu_usage_seconds_rate",     # rate, not cumulative counter
        "container_cpu_system_seconds_rate",    # rate
        "container_cpu_user_seconds_rate",       # rate
        "container_memory_usage_bytes",          # gauge
        "container_memory_working_set_bytes",    # gauge
        "container_memory_rss",                  # gauge
        "container_memory_cache",                # gauge
    ]
    WINDOW_SIZE = 30   # 30 x 15s = 7.5 minutes

    def __init__(self, model, scaler, pca, clip, threshold, metrics=None):
        self.model = model
        self.scaler = scaler
        self.pca = pca
        self.clip = clip
        self.threshold = threshold
        self.metrics = metrics or {}

    @classmethod
    def load(cls, path: str) -> "VAEAloneDetector":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt["model_config"]
        model = VAE(cfg["input_dim"], cfg["hidden1"], cfg["hidden2"], cfg["latent_dim"])
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return cls(
            model=model,
            scaler=ckpt["scaler"],
            pca=ckpt["pca"],
            clip=cfg["clip"],
            threshold=ckpt["threshold"],
            metrics=ckpt.get("eval_metrics"),
        )

    def score(self, raw_window_30x7: np.ndarray) -> dict:
        """raw_window_30x7: shape (30, 7), columns in FEATURE_COLS order.
        CPU columns must already be rate-converted (diff(value)/diff(timestamp),
        clipped at 0) if your source gives cumulative counters."""
        if raw_window_30x7.shape != (self.WINDOW_SIZE, len(self.FEATURE_COLS)):
            raise ValueError(
                f"expected shape ({self.WINDOW_SIZE}, {len(self.FEATURE_COLS)}), "
                f"got {raw_window_30x7.shape}"
            )
        scaled = np.clip(self.scaler.transform(raw_window_30x7), -self.clip, self.clip)
        flat = scaled.reshape(1, -1)
        pca_space = np.clip(self.pca.transform(flat), -self.clip, self.clip).astype(np.float32)

        with torch.no_grad():
            mse = self.model.anomaly_score(torch.from_numpy(pca_space)).item()

        return {
            "reconstruction_mse": mse,
            "is_anomaly": mse > self.threshold,
            "threshold_used": self.threshold,
        }


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "vae_cc1_final_model.pt"
    detector = VAEAloneDetector.load(path)
    print(f"Loaded VAE-alone detector from {path}")
    print(f"  threshold (val_p99): {detector.threshold:.5f}")
    print(f"  reported metrics: {detector.metrics}")

    dummy = np.random.randn(30, 7) * 0.1 + 0.5
    print("\nSmoke test:", detector.score(dummy))
