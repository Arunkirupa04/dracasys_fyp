"""Timestamp helpers for Prophet (requires timezone-naive datetimes)."""

from __future__ import annotations

import pandas as pd


def to_naive_utc(series: pd.Series | pd.DatetimeIndex) -> pd.Series:
    """
    Convert timestamps to timezone-naive UTC wall time for Prophet compatibility.

    Prophet does not support tz-aware ``ds`` columns. Inputs may be ISO-8601 with
    ``+00:00``; internally we normalise to naive UTC-equivalent datetimes.
    """
    if isinstance(series, pd.DatetimeIndex):
        s = pd.Series(series)
    else:
        s = pd.to_datetime(series, utc=True)
    if getattr(s.dt, "tz", None) is not None:
        s = pd.to_datetime(s.dt.strftime("%Y-%m-%d %H:%M:%S"))
    return s


def format_forecast_timestamp(ts: pd.Timestamp) -> str:
    """Format naive timestamp for API response (UTC suffix)."""
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.isoformat() + "+00:00"
