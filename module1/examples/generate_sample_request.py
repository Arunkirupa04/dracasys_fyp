#!/usr/bin/env python3
"""Generate a valid sample forecast request JSON (200 steps, 15-min interval)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

N = 200
HORIZON = 96
start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
timestamps = [
    (start + timedelta(minutes=15 * i)).isoformat()
    for i in range(N)
]
# Synthetic CPU: daily seasonality + noise
t = np.arange(N)
cpu = 20 + 10 * np.sin(2 * np.pi * t / 96) + np.random.default_rng(42).normal(0, 2, N)
cpu = np.clip(cpu, 0, 100)

payload = {
    "container_id": "c_1016",
    "historical_cpu": [round(float(x), 4) for x in cpu],
    "timestamps": timestamps,
    "sampling_interval_minutes": 15,
    "prediction_horizon_steps": HORIZON,
}

out = Path(__file__).parent / "sample_request.json"
out.write_text(json.dumps(payload, indent=2))
print(f"Wrote {out} ({N} history steps)")
