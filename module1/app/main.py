"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import register_exception_handlers, router
from app.config.settings import get_settings
from app.services.artifact_loader import load_artifacts
from app.services.forecast_service import ForecastOrchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    artifacts = load_artifacts(settings)
    app.state.orchestrator = ForecastOrchestrator(artifacts, settings)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="DracaSys Hybrid CPU Forecast Service",
        description=(
            "REST API exposing the frozen Hybrid Prophet+GRU production model "
            "for 24-hour CPU utilisation forecasting."
        ),
        version=settings.service_version,
        lifespan=lifespan,
    )
    register_exception_handlers(app)
    app.include_router(router)
    return app


app = create_app()
