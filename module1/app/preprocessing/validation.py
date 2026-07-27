"""Request validation and preprocessing."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from app.config.settings import Settings
from app.utils.errors import BadRequestError, UnprocessableEntityError
from app.utils.timestamps import to_naive_utc


def validate_forecast_request(
    container_id: str,
    historical_cpu: list[float],
    timestamps: list[str],
    sampling_interval_minutes: int,
    prediction_horizon_steps: int,
    settings: Settings,
) -> pd.DataFrame:
    """Validate input arrays and return a sorted history DataFrame."""
    if not container_id or not container_id.strip():
        raise BadRequestError("container_id is required")

    if len(historical_cpu) != len(timestamps):
        raise BadRequestError(
            "historical_cpu and timestamps must have the same length",
            {"cpu_len": len(historical_cpu), "timestamp_len": len(timestamps)},
        )

    if len(historical_cpu) < settings.min_history_steps:
        raise UnprocessableEntityError(
            f"Insufficient history: need at least {settings.min_history_steps} steps",
            {
                "provided": len(historical_cpu),
                "minimum": settings.min_history_steps,
                "recommended": settings.recommended_history_steps,
            },
        )

    if sampling_interval_minutes != settings.sampling_interval_minutes:
        raise BadRequestError(
            f"Unsupported sampling interval: {sampling_interval_minutes} minutes",
            {"supported": settings.sampling_interval_minutes},
        )

    if prediction_horizon_steps < 1 or prediction_horizon_steps > settings.max_horizon_steps:
        raise BadRequestError(
            f"prediction_horizon_steps must be between 1 and {settings.max_horizon_steps}",
            {"provided": prediction_horizon_steps},
        )

    cpu = np.asarray(historical_cpu, dtype=float)
    if np.any(~np.isfinite(cpu)):
        raise UnprocessableEntityError("historical_cpu contains NaN or infinite values")

    if np.any(cpu < settings.cpu_min_percent) or np.any(cpu > settings.cpu_max_percent):
        raise UnprocessableEntityError(
            "historical_cpu values must be within [0, 100] percent",
            {"min_allowed": settings.cpu_min_percent, "max_allowed": settings.cpu_max_percent},
        )

    if float(np.std(cpu)) < 1e-12:
        raise UnprocessableEntityError("historical_cpu has zero variance (degenerate series)")

    try:
        ts = to_naive_utc(pd.Series(timestamps))
    except Exception as exc:
        raise BadRequestError("Invalid timestamp format; use ISO-8601") from exc

    df = pd.DataFrame({"time_stamp": ts, "cpu": cpu}).sort_values("time_stamp")
    if df["time_stamp"].duplicated().any():
        raise UnprocessableEntityError("timestamps must be unique")

    deltas = df["time_stamp"].diff().dropna()
    expected = timedelta(minutes=sampling_interval_minutes)
    tolerance = timedelta(minutes=1)
    if not deltas.between(expected - tolerance, expected + tolerance).all():
        raise UnprocessableEntityError(
            "timestamps must be evenly spaced at the declared sampling interval",
            {"expected_minutes": sampling_interval_minutes},
        )

    return df.reset_index(drop=True)


def build_future_timestamps(
    last_timestamp: pd.Timestamp,
    horizon_steps: int,
    interval_minutes: int,
) -> pd.DatetimeIndex:
    """Generate future timestamps for the forecast horizon (naive UTC)."""
    if getattr(last_timestamp, "tzinfo", None) is not None:
        last_timestamp = last_timestamp.tz_localize(None)
    freq = f"{interval_minutes}min"
    start = last_timestamp + pd.Timedelta(minutes=interval_minutes)
    return pd.date_range(start=start, periods=horizon_steps, freq=freq)
