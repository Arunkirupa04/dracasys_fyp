"""Load and validate frozen production artifacts at startup."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config.settings import Settings
from app.preprocessing.scaling import load_scalers
from app.utils.errors import ServiceUnavailableError


@dataclass
class ArtifactBundle:
    """In-memory artifact bundle for inference."""

    model: Any
    scalers: dict
    res_mean: float
    res_std: float
    input_window: int
    forecast_horizon: int
    production_config: dict
    prophet_metadata: dict
    known_container_ids: list[str]
    model_version: str


def _load_json_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open() as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return [str(x) for x in payload]
    if isinstance(payload, dict) and "container_ids" in payload:
        return [str(x) for x in payload["container_ids"]]
    return []


def load_artifacts(settings: Settings) -> ArtifactBundle:
    """Load all artifacts; raise ServiceUnavailableError if any required file missing."""
    artifacts_dir = settings.artifacts_dir
    if not artifacts_dir.exists():
        raise ServiceUnavailableError(
            "Artifacts directory not found",
            {"path": str(artifacts_dir)},
        )

    for name, path in (
        ("model.keras", settings.model_path),
        ("scalers.pkl", settings.scalers_path),
        ("residual_stats.pkl", settings.residual_stats_path),
        ("production_config.json", settings.production_config_path),
    ):
        if not path.exists():
            raise ServiceUnavailableError(
                f"Required artifact missing: {name}",
                {"path": str(path)},
            )

    try:
        import tensorflow as tf

        model = tf.keras.models.load_model(str(settings.model_path))
    except Exception as exc:
        raise ServiceUnavailableError("Failed to load GRU model", {"error": str(exc)}) from exc

    scalers = load_scalers(settings.scalers_path)

    with settings.residual_stats_path.open("rb") as handle:
        residual_stats = pickle.load(handle)

    with settings.production_config_path.open() as handle:
        production_config = json.load(handle)

    prophet_metadata: dict = {}
    if settings.prophet_metadata_path.exists():
        with settings.prophet_metadata_path.open("rb") as handle:
            prophet_metadata = pickle.load(handle)

    known_ids = _load_json_list(settings.train_container_ids_path)
    if not known_ids:
        known_ids = sorted(scalers.keys())

    return ArtifactBundle(
        model=model,
        scalers=scalers,
        res_mean=float(residual_stats["res_mean"]),
        res_std=float(residual_stats["res_std"]),
        input_window=int(residual_stats.get("input_window", settings.input_window)),
        forecast_horizon=int(
            residual_stats.get("forecast_horizon", settings.default_horizon_steps)
        ),
        production_config=production_config,
        prophet_metadata=prophet_metadata,
        known_container_ids=known_ids,
        model_version=settings.model_version,
    )
