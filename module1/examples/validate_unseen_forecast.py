#!/usr/bin/env python3
"""Compare POST /forecast output against unseen validation ground truth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx
import numpy as np

DEFAULT_CONTAINER = "c_10312"
EXAMPLES = Path(__file__).parent


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container-id", default=DEFAULT_CONTAINER)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--request",
        type=Path,
        default=None,
        help="Forecast request JSON (default: unseen_<id>_forecast_request.json)",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="Validation JSON (default: unseen_<id>_validation_ground_truth.json)",
    )
    args = parser.parse_args()

    req_path = args.request or EXAMPLES / f"unseen_{args.container_id}_forecast_request.json"
    gt_path = args.ground_truth or EXAMPLES / f"unseen_{args.container_id}_validation_ground_truth.json"

    request = json.loads(req_path.read_text())
    ground_truth = json.loads(gt_path.read_text())

    with httpx.Client(base_url=args.base_url, timeout=120.0) as client:
        resp = client.post("/forecast", json=request)
        resp.raise_for_status()
        result = resp.json()

    actual = np.array(ground_truth["actual_cpu"], dtype=float)
    predicted = np.array(result["predicted_cpu_percent"], dtype=float)
    if len(actual) != len(predicted):
        print(
            f"Length mismatch: ground truth {len(actual)} vs prediction {len(predicted)}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Container: {args.container_id}")
    print(f"Scaler mode: {result.get('scaler_mode', 'n/a')}")
    print(f"History steps sent: {len(request['historical_cpu'])}")
    print(f"Horizon: {len(predicted)}")
    print(f"MAE:  {mae(actual, predicted):.4f}%")
    print(f"RMSE: {rmse(actual, predicted):.4f}%")
    print(f"MAPE: {mape(actual, predicted):.4f}%")


if __name__ == "__main__":
    main()
