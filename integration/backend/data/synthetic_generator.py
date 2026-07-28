"""
Generates the synthetic container metrics stream (Stream A) used by
Module 2 (resource forecasting). Module 3 no longer reads this stream —
it uses its own calibrated windows (m3_window_generator.py) matched to its
trained scaler's unit convention.

Layout:
  Rows  0 .. n_history-1        : normal history (pre-seeded GRU context)
  Rows  n_history .. end        : demo samples — a gentle load ramp begins
                                   partway through the demo tail, giving the
                                   "actual vs predicted" chart a visible trend
                                   for the GRU forecast to track.

Columns (7, in the order Module 2 expects):
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


def generate(n_history: int = 1000, n_demo: int = 10, seed: int = 42) -> pd.DataFrame:
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

    # Gentle load ramp over the back half of the demo tail — gives the
    # actual-vs-predicted chart a real trend without pushing the model far
    # outside its training distribution.
    ramp = np.zeros(total)
    ramp_start = n_history + max(1, n_demo // 3)
    if ramp_start < total:
        ramp_len = total - ramp_start
        ramp[ramp_start:] = np.linspace(0, 0.5, ramp_len)
    cpu_rate = np.clip(cpu_rate + ramp, 0.05, 2.2)

    # Cumulative counters: integrate rate × Δt (15 s per tick)
    # Offset so the middle of the stream aligns with training mean ≈ 1050 s
    cpu_total  = np.cumsum(cpu_rate * 15.0) + 400.0
    cpu_system = np.cumsum(cpu_rate * 0.33 * 15.0 + 0.01 * rng.standard_normal(total)) + 44.0
    cpu_user   = np.cumsum(cpu_rate * 0.65 * 15.0 + 0.01 * rng.standard_normal(total)) + 120.0

    # ── Memory gauges (bytes) ──────────────────────────────────────────
    mem_base = 82_000_000   # ~82 MB baseline (matches training mean)
    mem_ramp = np.zeros(total)
    if ramp_start < total:
        mem_ramp[ramp_start:] = np.linspace(0, 18_000_000, total - ramp_start)

    mem_usage = (
        mem_base
        + 12_000_000 * np.sin(2 * np.pi * t / 800)
        + 4_000_000  * rng.standard_normal(total)
        + mem_ramp
    )
    mem_wss   = 0.93 * mem_usage + 1_500_000 * rng.standard_normal(total)
    mem_rss   = 0.87 * mem_usage + 1_500_000 * rng.standard_normal(total)
    mem_cache = 0.10 * mem_usage + 800_000   * rng.standard_normal(total)

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
    return df   # shape: (n_history + n_demo, 7)
