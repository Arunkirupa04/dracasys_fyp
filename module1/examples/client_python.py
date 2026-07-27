#!/usr/bin/env python3
"""Example: call the forecast API from Python."""

from __future__ import annotations

import json
from pathlib import Path

import httpx

URL = "http://localhost:8000/forecast"
sample = json.loads((Path(__file__).parent / "sample_request.json").read_text())

response = httpx.post(URL, json=sample, timeout=120.0)
response.raise_for_status()
data = response.json()
print("Status:", data["status"])
print("First 3 forecast CPU values:", data["forecast"]["predicted_cpu_percent"][:3])
print("Processing time (ms):", data["metadata"]["processing_time_ms"])
