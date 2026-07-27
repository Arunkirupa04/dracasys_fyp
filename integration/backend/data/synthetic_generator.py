"""
Generates the synthetic container metrics stream (Stream A) used by
Module 2 (resource forecasting) and Module 3 (system anomaly detection).

Stream A layout:
  Rows  0 – 999   : normal history (pre-seeded context for M2 + M3)
  Rows 1000 – 1000: demo sample 1 — normal
  Rows 1001 – 1001: demo sample 2 — normal
  Rows 1002 – 1002: demo sample 3 — MEMORY LEAK anomaly  → triggers M3
  Rows 1003 – 1003: demo sample 4 — CPU SATURATION + elevated memory → triggers M3

Columns (7, in the order Module 2 and Module 3 both expect):
  container_cpu_usage_seconds_total    (cumulative counter)
  container_cpu_system_seconds_total   (cumulative counter)
  container_cpu_user_seconds_total     (cumulative counter)
  container_memory_usage_bytes         (gauge)
  container_memory_working_set_bytes   (gauge)
  container_memory_rss                 (gauge)
  container_memory_cache               (gauge)

Target distribution is calibrated to match normalization_stats.json:
  cpu_total mean ≈ 1050, std ≈ 1100  (cumulative, so middle of the stream)
  memory_usage mean ≈ 82.7 MB, std ≈ 16 MB
"""

import numpy as np
import pandas as pd

COLUMNS = [
    "container_cpu_usage_seconds_total",
    "container_cpu_system_seconds_total",
    "container_cpu_user_seconds_total",
    "container_memory_usage_bytes",
    "container_memory_working_set_bytes",
    "container_memory_rss",
    "container_memory_cache",
]

# Anomaly magnitudes (tuned so M3 reconstruction error > threshold 0.0980)
MEMORY_LEAK_DELTA   = 900_000_000   # +900 MB
CPU_SATURATION_RATE = 4.5           # 4.5× normal rate


def generate(n_history: int = 1000, n_demo: int = 4, seed: int = 42) -> pd.DataFrame:
    """
    Returns DataFrame of shape (n_history + n_demo, 7).
    All values are raw (not normalized).
    """
    rng   = np.random.default_rng(seed)
    total = n_history + n_demo
    t     = np.arange(total, dtype=np.float64)

    # ── CPU rates (realistic workload with daily cycle) ────────────────
    cpu_rate = (
        0.40
        + 0.22 * np.sin(2 * np.pi * t / 600)          # ~2.5-hour cycle
        + 0.08 * np.sin(2 * np.pi * t / 120)           # short burst cycle
        + 0.05 * rng.standard_normal(total)
    )
    cpu_rate = np.clip(cpu_rate, 0.05, 1.8)

    # Cumulative counters: integrate rate × Δt (15 s per tick)
    # Offset so the middle of the stream aligns with training mean ≈ 1050 s
    cpu_total  = np.cumsum(cpu_rate * 15.0) + 400.0
    cpu_system = np.cumsum(cpu_rate * 0.33 * 15.0 + 0.01 * rng.standard_normal(total)) + 44.0
    cpu_user   = np.cumsum(cpu_rate * 0.65 * 15.0 + 0.01 * rng.standard_normal(total)) + 120.0

    # ── Memory gauges (bytes) ──────────────────────────────────────────
    mem_base = 82_000_000   # ~82 MB baseline (matches training mean)
    mem_usage = (
        mem_base
        + 12_000_000 * np.sin(2 * np.pi * t / 800)
        + 4_000_000  * rng.standard_normal(total)
    )
    mem_wss   = 0.93 * mem_usage + 1_500_000 * rng.standard_normal(total)
    mem_rss   = 0.87 * mem_usage + 1_500_000 * rng.standard_normal(total)
    mem_cache = 0.10 * mem_usage + 800_000   * rng.standard_normal(total)

    # ── Anomaly injection into demo samples 3 and 4 ───────────────────
    # Sample 3 (index n_history + 2 = 1002): memory leak spike
    i3 = n_history + 2
    mem_usage[i3] += MEMORY_LEAK_DELTA
    mem_wss[i3]   += int(MEMORY_LEAK_DELTA * 0.97)
    mem_rss[i3]   += int(MEMORY_LEAK_DELTA * 0.90)

    # Sample 4 (index 1003): CPU saturation + memory still elevated
    i4 = n_history + 3
    # Spike the last 5 ticks of CPU rate (affects cumulative counter at i4)
    cpu_rate[i4 - 4 : i4 + 1] = CPU_SATURATION_RATE
    cpu_total[i4 - 4 : i4 + 1] = np.cumsum(
        cpu_rate[i4 - 4 : i4 + 1] * 15.0
    ) + cpu_total[i4 - 5]
    mem_usage[i4] += int(MEMORY_LEAK_DELTA * 0.85)
    mem_wss[i4]   += int(MEMORY_LEAK_DELTA * 0.82)

    # Clip to physically plausible ranges
    mem_usage  = np.clip(mem_usage,  10_000_000, 4_000_000_000)
    mem_wss    = np.clip(mem_wss,     8_000_000, 4_000_000_000)
    mem_rss    = np.clip(mem_rss,     5_000_000, 4_000_000_000)
    mem_cache  = np.clip(mem_cache,     500_000,   500_000_000)
    cpu_total  = np.abs(cpu_total)
    cpu_system = np.abs(cpu_system)
    cpu_user   = np.abs(cpu_user)

    df = pd.DataFrame(
        {
            "container_cpu_usage_seconds_total":   cpu_total,
            "container_cpu_system_seconds_total":  cpu_system,
            "container_cpu_user_seconds_total":    cpu_user,
            "container_memory_usage_bytes":        mem_usage,
            "container_memory_working_set_bytes":  mem_wss,
            "container_memory_rss":                mem_rss,
            "container_memory_cache":              mem_cache,
        }
    )
    return df   # shape: (1004, 7)


def describe_sample(df: pd.DataFrame, sample_index: int, n_history: int = 1000) -> str:
    """Human-readable description of a demo sample (0-based)."""
    row_idx = n_history + sample_index
    row = df.iloc[row_idx]
    labels = ["Normal workload", "Normal workload", "Memory leak anomaly", "CPU saturation + memory spike"]
    return labels[sample_index]
