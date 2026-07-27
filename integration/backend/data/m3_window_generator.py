"""
Generates 30-row windows for Module 3 (VAE system anomaly detector).

M3's bundled RobustScaler was fitted on AIOpsArena data where memory values
are in GB scale (center_ ≈ 0.793, scale_ ≈ 0.45) and CPU rates are near 0
(center_ ≈ 0, scale_ ≈ 1.0). This is a different unit convention from M2.

We generate data calibrated directly to M3's scaler parameters so that:
  - Normal windows produce reconstruction_mse ≈ 0.02  (well below threshold 0.098)
  - Anomalous windows produce reconstruction_mse >> 0.098

Column order (must match M3 handover §2):
  [cpu_rate, cpu_sys_rate, cpu_usr_rate, mem_usage, mem_wss, mem_rss, mem_cache]
  Units: CPU in Hz (small, ~0-0.01), memory in GB
"""
import numpy as np

# From the bundled RobustScaler (inspected):
SCALER_CENTER = np.array([0.0, 0.0, 0.0, 0.793, 0.793, 0.762, 0.455])
SCALER_SCALE  = np.array([1.0, 1.0, 1.0, 0.448, 0.453, 0.453, 0.448])
CLIP = 20.0   # M3 uses ±20 clip before and after PCA


def _normal_row(rng: np.random.Generator) -> np.ndarray:
    """
    Generate one row of normal container behavior in M3's expected units.
    Targets scaled values in roughly [-0.5, 0.5] for all features.
    raw = scaled * scale + center
    """
    # CPU rates: target scaled ≈ N(0, 0.3) → raw = 0 + N(0,0.3)*1.0
    cpu    = rng.normal(0.0,  0.003, 3)    # tiny CPU rates, near 0
    cpu    = np.clip(cpu, 0.0, 0.05)       # non-negative

    # Memory: target scaled ≈ N(0, 0.3) → raw ≈ center + N(0,0.3)*scale
    mem_u  = SCALER_CENTER[3] + rng.normal(0.0, 0.08) * SCALER_SCALE[3]
    mem_w  = SCALER_CENTER[4] + rng.normal(0.0, 0.08) * SCALER_SCALE[4]
    mem_r  = SCALER_CENTER[5] + rng.normal(0.0, 0.08) * SCALER_SCALE[5]
    mem_c  = SCALER_CENTER[6] + rng.normal(0.0, 0.08) * SCALER_SCALE[6]
    mem    = np.array([mem_u, mem_w, mem_r, mem_c])
    mem    = np.clip(mem, 0.01, 4.0)

    return np.concatenate([cpu, mem])


def _anomalous_row_memory_leak(rng: np.random.Generator) -> np.ndarray:
    """
    Memory leak: mem values shifted to ~+3 IQR above normal → scaled ≈ +3.
    CPU stays normal. This pushes reconstruction error well above threshold.
    """
    cpu = np.clip(rng.normal(0.0, 0.003, 3), 0.0, 0.05)

    # Push memory to 3× IQR above center → scaled ≈ 3
    mem_u = SCALER_CENTER[3] + 3.0 * SCALER_SCALE[3] + rng.normal(0, 0.02)
    mem_w = SCALER_CENTER[4] + 3.0 * SCALER_SCALE[4] + rng.normal(0, 0.02)
    mem_r = SCALER_CENTER[5] + 2.8 * SCALER_SCALE[5] + rng.normal(0, 0.02)
    mem_c = SCALER_CENTER[6] + 0.5 * SCALER_SCALE[6] + rng.normal(0, 0.01)
    mem   = np.clip(np.array([mem_u, mem_w, mem_r, mem_c]), 0.01, 10.0)

    return np.concatenate([cpu, mem])


def _anomalous_row_cpu_saturated(rng: np.random.Generator) -> np.ndarray:
    """
    CPU saturation: CPU rate spikes to ~+4 IQR, memory also elevated.
    """
    # CPU scaled ≈ +4 → raw = 0 + 4*1.0 = 4.0 Hz
    cpu = np.array([
        4.0 + rng.normal(0, 0.2),
        1.5 + rng.normal(0, 0.1),
        2.5 + rng.normal(0, 0.1),
    ])
    cpu = np.clip(cpu, 0.0, 10.0)

    mem_u = SCALER_CENTER[3] + 2.5 * SCALER_SCALE[3] + rng.normal(0, 0.02)
    mem_w = SCALER_CENTER[4] + 2.5 * SCALER_SCALE[4] + rng.normal(0, 0.02)
    mem_r = SCALER_CENTER[5] + 2.2 * SCALER_SCALE[5] + rng.normal(0, 0.02)
    mem_c = SCALER_CENTER[6] + 0.3 * SCALER_SCALE[6] + rng.normal(0, 0.01)
    mem   = np.clip(np.array([mem_u, mem_w, mem_r, mem_c]), 0.01, 10.0)

    return np.concatenate([cpu, mem])


def generate_windows(seed: int = 77) -> list[np.ndarray]:
    """
    Returns a list of 4 windows, each shape (30, 7) float32.
      Window 0 — normal
      Window 1 — normal
      Window 2 — memory leak anomaly
      Window 3 — CPU saturation + memory anomaly
    """
    rng = np.random.default_rng(seed)
    windows = []

    for desc, row_fn in [
        ("normal",   _normal_row),
        ("normal",   _normal_row),
        ("mem_leak", _anomalous_row_memory_leak),
        ("cpu_sat",  _anomalous_row_cpu_saturated),
    ]:
        rows = np.stack([row_fn(rng) for _ in range(30)], axis=0)  # (30, 7)
        windows.append(rows.astype(np.float32))

    return windows
