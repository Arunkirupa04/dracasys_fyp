"""HTTP and validation exceptions."""

from __future__ import annotations


class ForecastServiceError(Exception):
    """Base service error with HTTP mapping."""

    status_code: int = 500
    error_code: str = "internal_error"

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class BadRequestError(ForecastServiceError):
    status_code = 400
    error_code = "bad_request"


class NotFoundError(ForecastServiceError):
    status_code = 404
    error_code = "not_found"


class UnprocessableEntityError(ForecastServiceError):
    status_code = 422
    error_code = "validation_error"


class ServiceUnavailableError(ForecastServiceError):
    status_code = 503
    error_code = "service_unavailable"
