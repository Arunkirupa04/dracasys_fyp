"""Pydantic request/response schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ForecastRequest(BaseModel):
    """Single-container forecast request."""

    container_id: str = Field(
        ...,
        description="Container identifier. Known IDs use frozen training scalers.",
        examples=["c_1016"],
    )
    historical_cpu: list[float] = Field(
        ...,
        description="Historical CPU utilisation in percent, oldest first.",
        min_length=1,
    )
    timestamps: list[str] = Field(
        ...,
        description="ISO-8601 UTC timestamps aligned with historical_cpu.",
        min_length=1,
    )
    sampling_interval_minutes: int = Field(
        default=15,
        description="Minutes between consecutive samples (must be 15).",
    )
    prediction_horizon_steps: int = Field(
        default=96,
        ge=1,
        le=96,
        description="Forecast length in steps (96 = 24 hours at 15-min sampling).",
    )

    @field_validator("historical_cpu")
    @classmethod
    def cpu_not_empty(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("historical_cpu cannot be empty")
        return v


class ForecastMetadata(BaseModel):
    model_version: str
    processing_time_ms: float
    scaler_mode: Literal["known", "new"]
    history_steps_used: int
    horizon_steps: int
    sampling_interval_minutes: int


class ForecastData(BaseModel):
    timestamps: list[str]
    predicted_cpu_percent: list[float]
    prophet_component_percent: list[float]
    gru_residual_component_percent: list[float]


class ForecastResponse(BaseModel):
    status: Literal["success"] = "success"
    container_id: str
    metadata: ForecastMetadata
    forecast: ForecastData


class BatchForecastRequest(BaseModel):
    requests: list[ForecastRequest] = Field(..., min_length=1, max_length=50)


class BatchForecastItem(BaseModel):
    status: str
    container_id: str | None = None
    processing_time_ms: float | None = None
    forecast: ForecastData | None = None
    error: str | None = None


class BatchForecastResponse(BaseModel):
    status: Literal["success"] = "success"
    results: list[BatchForecastItem]
    total: int
    succeeded: int
    failed: int


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    model_loaded: bool
    model_version: str | None = None
    artifacts_dir: str


class ModelInfoResponse(BaseModel):
    model_version: str
    model_type: str
    input_window: int
    default_horizon_steps: int
    max_horizon_steps: int
    sampling_interval_minutes: int
    known_containers_count: int
    residual_normalization: str
    prophet: dict[str, Any]
    architecture_summary: str
    minimum_history_steps: int = 200
    recommended_history_steps: int = 672


class VersionResponse(BaseModel):
    service_version: str
    model_version: str
    api_version: str = "v1"


class ErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    error_code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
