"""Service configuration — all parameters externalised."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="FORECAST_",
        env_file=".env",
        extra="ignore",
    )

    service_name: str = "dracasys-hybrid-forecast"
    service_version: str = "1.0.0"
    model_version: str = "hybrid_v1"

    artifacts_dir: Path = Field(
        default=Path(__file__).resolve().parents[2] / "artifacts" / "hybrid_v1",
    )

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    input_window: int = 96
    default_horizon_steps: int = 96
    max_horizon_steps: int = 96
    sampling_interval_minutes: int = 15

    min_history_steps: int = 200
    recommended_history_steps: int = 672

    cpu_min_percent: float = 0.0
    cpu_max_percent: float = 100.0

    prophet_daily_seasonality: bool = True
    prophet_weekly_seasonality: bool = False

    @property
    def model_path(self) -> Path:
        return self.artifacts_dir / "model.keras"

    @property
    def scalers_path(self) -> Path:
        return self.artifacts_dir / "scalers.pkl"

    @property
    def residual_stats_path(self) -> Path:
        return self.artifacts_dir / "residual_stats.pkl"

    @property
    def production_config_path(self) -> Path:
        return self.artifacts_dir / "production_config.json"

    @property
    def prophet_metadata_path(self) -> Path:
        return self.artifacts_dir / "prophet_metadata.pkl"

    @property
    def train_container_ids_path(self) -> Path:
        return self.artifacts_dir / "train_container_ids.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
