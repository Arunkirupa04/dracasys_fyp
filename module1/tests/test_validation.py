"""Validation tests — no model loading required."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.config.settings import Settings
from app.preprocessing.validation import validate_forecast_request
from app.utils.errors import BadRequestError, UnprocessableEntityError


def _make_series(n: int = 200) -> tuple[list[float], list[str]]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ts = [
        (start + timedelta(minutes=15 * i)).isoformat()
        for i in range(n)
    ]
    cpu = [float(20 + (i % 96) * 0.1) for i in range(n)]
    return cpu, ts


@pytest.fixture
def settings() -> Settings:
    return Settings()


def test_valid_request(settings: Settings) -> None:
    cpu, ts = _make_series(200)
    df = validate_forecast_request(
        "c_test", cpu, ts, 15, 96, settings,
    )
    assert len(df) == 200


def test_insufficient_history(settings: Settings) -> None:
    cpu, ts = _make_series(50)
    with pytest.raises(UnprocessableEntityError):
        validate_forecast_request("c_test", cpu, ts, 15, 96, settings)


def test_mismatched_lengths(settings: Settings) -> None:
    cpu, ts = _make_series(200)
    with pytest.raises(BadRequestError):
        validate_forecast_request("c_test", cpu[:-1], ts, 15, 96, settings)


def test_invalid_interval(settings: Settings) -> None:
    cpu, ts = _make_series(200)
    with pytest.raises(BadRequestError):
        validate_forecast_request("c_test", cpu, ts, 5, 96, settings)


def test_nan_cpu(settings: Settings) -> None:
    cpu, ts = _make_series(200)
    cpu[0] = float("nan")
    with pytest.raises(UnprocessableEntityError):
        validate_forecast_request("c_test", cpu, ts, 15, 96, settings)
