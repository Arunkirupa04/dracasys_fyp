#!/usr/bin/env python3
"""Build unseen-container forecast request + ground-truth validation JSON files.

Reads resampled unseen container data from the repo (not bundled in forecast_service).
Outputs two files under forecast_service/examples/:

  unseen_<container_id>_forecast_request.json   — history only (excludes last 96 steps)
  unseen_<container_id>_validation_ground_truth.json — actual next 96 steps for MAE checks

Usage (from repo root):
  python forecast_service/examples/generate_unseen_validation_example.py
  python forecast_service/examples/generate_unseen_validation_example.py --container-id c_11013
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

DEFAULT_CONTAINER = "c_10312"
HORIZON = 96
SAMPLING = 15
MIN_HISTORY = 200
BASE_TS = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)


def _timestamps(n: int, offset: int = 0) -> list[str]:
    return [
        (BASE_TS + timedelta(minutes=SAMPLING * (offset + i))).strftime(
            "%Y-%m-%dT%H:%M:%S+00:00"
        )
        for i in range(n)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container-id", default=DEFAULT_CONTAINER)
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "production/hybrid/cache/unseen_resampled.parquet",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent,
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    group = (
        df[df["container_id"] == args.container_id]
        .sort_values("time_stamp")
        .reset_index(drop=True)
    )
    if group.empty:
        raise SystemExit(f"Container {args.container_id!r} not found in {args.parquet}")

    n = len(group)
    if n < MIN_HISTORY + HORIZON:
        raise SystemExit(
            f"Need >= {MIN_HISTORY + HORIZON} rows for {args.container_id}, got {n}"
        )

    history = group.iloc[:-HORIZON]
    validation = group.iloc[-HORIZON:]
    hist_ts = _timestamps(len(history))
    val_ts = _timestamps(HORIZON, offset=len(history))

    request = {
        "container_id": args.container_id,
        "historical_cpu": [round(float(x), 4) for x in history["cpu_usage"]],
        "timestamps": hist_ts,
        "sampling_interval_minutes": SAMPLING,
        "prediction_horizon_steps": HORIZON,
    }

    ground_truth = {
        "container_id": args.container_id,
        "description": (
            f"Ground-truth CPU for the next {HORIZON} steps (24 h @ 15 min) immediately "
            f"following the history in unseen_{args.container_id}_forecast_request.json. "
            "Compare against POST /forecast predictions (MAE, RMSE, MAPE)."
        ),
        "actual_cpu": [round(float(x), 4) for x in validation["cpu_usage"]],
        "timestamps": val_ts,
        "sampling_interval_minutes": SAMPLING,
        "history_steps": len(history),
        "validation_steps": HORIZON,
        "history_last_timestamp": hist_ts[-1],
        "forecast_first_timestamp": val_ts[0],
        "source": "production/hybrid/cache/unseen_resampled.parquet (Alibaba unseen holdout)",
        "scaler_mode_expected": "unseen",
    }

    req_path = args.out_dir / f"unseen_{args.container_id}_forecast_request.json"
    val_path = args.out_dir / f"unseen_{args.container_id}_validation_ground_truth.json"
    req_path.write_text(json.dumps(request, indent=2) + "\n")
    val_path.write_text(json.dumps(ground_truth, indent=2) + "\n")

    print(f"Wrote {req_path} ({len(history)} history steps)")
    print(f"Wrote {val_path} ({HORIZON} validation steps)")


if __name__ == "__main__":
    main()
