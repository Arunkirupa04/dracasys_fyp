"""API schema tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.api.schemas import ForecastRequest


def test_forecast_request_defaults() -> None:
    req = ForecastRequest(
        container_id="c_1",
        historical_cpu=[1.0] * 10,
        timestamps=["2026-01-01T00:00:00+00:00"] * 10,
    )
    assert req.prediction_horizon_steps == 96
    assert req.sampling_interval_minutes == 15


def test_horizon_max() -> None:
    with pytest.raises(ValidationError):
        ForecastRequest(
            container_id="c_1",
            historical_cpu=[1.0],
            timestamps=["2026-01-01T00:00:00+00:00"],
            prediction_horizon_steps=200,
        )
