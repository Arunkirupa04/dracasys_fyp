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


def _anomalous_row_severe_combined(rng: np.random.Generator) -> np.ndarray:
    """
    Severe combined degradation: both CPU saturation and memory leak at once —
    used as the "grand finale" anomaly, worse than either alone.
    """
    cpu = np.array([
        5.5 + rng.normal(0, 0.2),
        2.2 + rng.normal(0, 0.1),
        3.3 + rng.normal(0, 0.1),
    ])
    cpu = np.clip(cpu, 0.0, 12.0)

    mem_u = SCALER_CENTER[3] + 4.0 * SCALER_SCALE[3] + rng.normal(0, 0.03)
    mem_w = SCALER_CENTER[4] + 4.0 * SCALER_SCALE[4] + rng.normal(0, 0.03)
    mem_r = SCALER_CENTER[5] + 3.6 * SCALER_SCALE[5] + rng.normal(0, 0.03)
    mem_c = SCALER_CENTER[6] + 0.6 * SCALER_SCALE[6] + rng.normal(0, 0.01)
    mem   = np.clip(np.array([mem_u, mem_w, mem_r, mem_c]), 0.01, 12.0)

    return np.concatenate([cpu, mem])


# Demo narrative across the streamed samples (0-based sample index → anomaly
# generator). Any index not listed here is normal. Designed to loosely
# correlate with Module 4's single confirmed security alert (see
# module4_detector.ALERT_SAMPLE_INDEX): a CPU spike (index 6) precedes it,
# and a severe combined degradation (index 9) follows as the aftermath.
ANOMALY_PLAN = {
    3: ("mem_leak", _anomalous_row_memory_leak),
    6: ("cpu_sat",  _anomalous_row_cpu_saturated),
    9: ("severe",   _anomalous_row_severe_combined),
}


def generate_windows(n: int = 10, seed: int = 77) -> list[np.ndarray]:
    """
    Returns a list of n windows, each shape (30, 7) float32.
    All indices are normal except those in ANOMALY_PLAN.
    """
    rng = np.random.default_rng(seed)
    windows = []

    for i in range(n):
        _, row_fn = ANOMALY_PLAN.get(i, ("normal", _normal_row))
        rows = np.stack([row_fn(rng) for _ in range(30)], axis=0)  # (30, 7)
        windows.append(rows.astype(np.float32))

    return windows
