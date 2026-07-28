# API Documentation

Documentation for **integrating with the DracaSys Hybrid CPU Forecast REST API**. No knowledge of Prophet, GRU, or internal pipeline code is required.

## Quick start

1. Ensure the service is running (see [implementation/deployment.md](../implementation/deployment.md) if you are hosting it).
2. Read [integration.md](integration.md) for the end-to-end flow.
3. Use [reference.md](reference.md) and [schemas.md](schemas.md) for exact request/response fields.

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d @examples/sample_request.json
```

Interactive OpenAPI UI: **http://localhost:8000/docs**

## Contents

| Document | Description |
|----------|-------------|
| [overview.md](overview.md) | API design, endpoints summary, content types |
| [integration.md](integration.md) | **Start here** — how other modules call the API |
| [reference.md](reference.md) | Full endpoint reference with examples |
| [schemas.md](schemas.md) | Request/response field definitions and validation rules |
| [errors.md](errors.md) | HTTP status codes, error codes, troubleshooting |
| [examples.md](examples.md) | Sample payloads, client code, validation holdout |
| [faq.md](faq.md) | Common API and integration questions |

## Related

- Client code: [examples/](../../examples/)
- Service deployment: [implementation/deployment.md](../implementation/deployment.md)
- Model limits: [implementation/limitations.md](../implementation/limitations.md)
