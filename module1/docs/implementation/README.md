# Implementation Documentation

Documentation for **deploying, maintaining, and extending** the forecast service. Intended for operators and contributors working on the codebase — not required for API consumers.

## Getting started

| Document | Description |
|----------|-------------|
| [deployment.md](deployment.md) | Install, configure, run, Docker |
| [developer_guide.md](developer_guide.md) | Local development workflow |
| [testing.md](testing.md) | Test strategy and release checklist |

## Architecture & pipeline

| Document | Description |
|----------|-------------|
| [architecture.md](architecture.md) | High-level system architecture |
| [system_design.md](system_design.md) | Components and responsibilities |
| [forecast_pipeline.md](forecast_pipeline.md) | End-to-end inference pipeline |
| [preprocessing.md](preprocessing.md) | Input validation and scaling |
| [prophet_pipeline.md](prophet_pipeline.md) | Prophet seasonal component |
| [gru_pipeline.md](gru_pipeline.md) | GRU residual component |

## Model & configuration

| Document | Description |
|----------|-------------|
| [artifact_reference.md](artifact_reference.md) | Frozen artifact files |
| [configuration.md](configuration.md) | Environment variables and settings |
| [limitations.md](limitations.md) | Known limits |
| [versioning.md](versioning.md) | Version policy |

## Reference

| Document | Description |
|----------|-------------|
| [faq.md](faq.md) | Deployment and model questions |
| [glossary.md](glossary.md) | Term definitions |
| [future_extensions.md](future_extensions.md) | Planned extensions |

## API consumers

If you only need to **call** the service, use **[docs/api/](../api/README.md)** instead.
