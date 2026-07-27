# API Overview

REST API built with **FastAPI**. All forecast endpoints accept and return **JSON**.

## Design principles

- **Synchronous** — client waits for forecast (Prophet fit is on-request)
- **Stateless** — no session; full history sent each time
- **Explicit validation** — reject bad input before inference
- **OpenAPI** — Swagger UI at `/docs`

## Endpoints summary

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Entry point |
| GET | `/health` | Liveness / model loaded |
| GET | `/version` | Version strings |
| GET | `/model/info` | Capabilities and constraints |
| POST | `/forecast` | One container |
| POST | `/forecast/batch` | Up to 50 containers |

## Typical integration flow

1. `GET /health` — confirm service ready
2. `GET /model/info` — read `minimum_history_steps`, `sampling_interval_minutes`
3. `POST /forecast` — send history, receive 96-step forecast

## Content type

- Request: `Content-Type: application/json`
- Response: `application/json`

## Rate limiting

Not built-in. Deploy API gateway if needed.

## Authentication

Not built-in. Place behind API gateway or add FastAPI dependency for production.

See [integration.md](integration.md) and [reference.md](reference.md).
