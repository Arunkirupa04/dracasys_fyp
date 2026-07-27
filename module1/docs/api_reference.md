# API Reference

Base URL: `http://<host>:<port>` (default `http://localhost:8000`)

## GET /

Service discovery.

**Response 200:**

```json
{
  "service": "dracasys-hybrid-forecast",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health"
}
```

---

## GET /health

**Response 200:**

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `healthy` \| `unhealthy` |
| `model_loaded` | bool | Artifacts loaded successfully |
| `model_version` | string \| null | e.g. `hybrid_v1` |
| `artifacts_dir` | string | Path to artifact bundle |

---

## GET /version

**Response 200:**

| Field | Description |
|-------|-------------|
| `service_version` | API package version |
| `model_version` | Frozen model bundle ID |
| `api_version` | `v1` |

---

## GET /model/info

**Response 200:** Model metadata including `input_window` (96), `known_containers_count` (435), Prophet config, architecture summary.

---

## POST /forecast

Single-container 24-hour forecast.

**Request body:** See [request_response_spec.md](request_response_spec.md)

**Response 200:** ForecastResponse

**Errors:** 400, 422, 503

---

## POST /forecast/batch

Up to 50 forecasts per request. Partial success allowed.

**Response 200:** BatchForecastResponse

---

## OpenAPI

Auto-generated at `/docs` and `/openapi.json`.
