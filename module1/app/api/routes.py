"""FastAPI route definitions."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.api.schemas import (
    BatchForecastRequest,
    BatchForecastResponse,
    BatchForecastItem,
    ErrorResponse,
    ForecastData,
    ForecastMetadata,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    ModelInfoResponse,
    VersionResponse,
)
from app.config.settings import get_settings
from app.utils.errors import ForecastServiceError

router = APIRouter()


def _orchestrator(request: Request):
    return request.app.state.orchestrator


@router.get("/", tags=["Root"])
def root() -> dict[str, str]:
    settings = get_settings()
    return {
        "service": settings.service_name,
        "version": settings.service_version,
        "docs": "/docs",
        "health": "/health",
    }


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health(request: Request) -> HealthResponse:
    settings = get_settings()
    orch = getattr(request.app.state, "orchestrator", None)
    loaded = orch is not None and orch.is_ready
    return HealthResponse(
        status="healthy" if loaded else "unhealthy",
        model_loaded=loaded,
        model_version=settings.model_version if loaded else None,
        artifacts_dir=str(settings.artifacts_dir),
    )


@router.get("/version", response_model=VersionResponse, tags=["Meta"])
def version() -> VersionResponse:
    settings = get_settings()
    return VersionResponse(
        service_version=settings.service_version,
        model_version=settings.model_version,
    )


@router.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
def model_info(request: Request) -> ModelInfoResponse:
    orch = _orchestrator(request)
    info = orch.model_info()
    settings = get_settings()
    return ModelInfoResponse(
        **info,
        minimum_history_steps=settings.min_history_steps,
        recommended_history_steps=settings.recommended_history_steps,
    )


@router.post(
    "/forecast",
    response_model=ForecastResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    tags=["Forecast"],
)
def forecast(request: Request, body: ForecastRequest) -> ForecastResponse:
    orch = _orchestrator(request)
    result, elapsed_ms = orch.forecast(
        container_id=body.container_id,
        historical_cpu=body.historical_cpu,
        timestamps=body.timestamps,
        sampling_interval_minutes=body.sampling_interval_minutes,
        prediction_horizon_steps=body.prediction_horizon_steps,
    )
    settings = get_settings()
    return ForecastResponse(
        container_id=result.container_id,
        metadata=ForecastMetadata(
            model_version=settings.model_version,
            processing_time_ms=round(elapsed_ms, 2),
            scaler_mode=result.scaler_mode,  # type: ignore[arg-type]
            history_steps_used=result.history_steps,
            horizon_steps=result.horizon_steps,
            sampling_interval_minutes=result.sampling_interval_minutes,
        ),
        forecast=ForecastData(
            timestamps=result.forecast_timestamps,
            predicted_cpu_percent=result.predicted_cpu,
            prophet_component_percent=result.prophet_cpu,
            gru_residual_component_percent=result.residual_cpu,
        ),
    )


@router.post(
    "/forecast/batch",
    response_model=BatchForecastResponse,
    tags=["Forecast"],
)
def forecast_batch(request: Request, body: BatchForecastRequest) -> BatchForecastResponse:
    orch = _orchestrator(request)
    raw = [
        {
            "container_id": r.container_id,
            "historical_cpu": r.historical_cpu,
            "timestamps": r.timestamps,
            "sampling_interval_minutes": r.sampling_interval_minutes,
            "prediction_horizon_steps": r.prediction_horizon_steps,
        }
        for r in body.requests
    ]
    results_raw = orch.forecast_batch(raw)
    results = []
    for item in results_raw:
        if item["status"] == "success":
            fc = item["forecast"]
            results.append(BatchForecastItem(
                status="success",
                container_id=item["container_id"],
                processing_time_ms=item["processing_time_ms"],
                forecast=ForecastData(**fc),
            ))
        else:
            results.append(BatchForecastItem(
                status="error",
                container_id=item.get("container_id"),
                error=item.get("error"),
            ))
    succeeded = sum(1 for r in results if r.status == "success")
    return BatchForecastResponse(
        results=results,
        total=len(results),
        succeeded=succeeded,
        failed=len(results) - succeeded,
    )


def register_exception_handlers(app) -> None:
    @app.exception_handler(ForecastServiceError)
    async def forecast_error_handler(_request: Request, exc: ForecastServiceError):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error_code=exc.error_code,
                message=exc.message,
                details=exc.details,
            ).model_dump(),
        )
