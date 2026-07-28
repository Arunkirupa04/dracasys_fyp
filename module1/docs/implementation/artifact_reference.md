# Artifact Reference

All files under `artifacts/hybrid_v1/`. **Read-only at runtime. Never retrain or overwrite in deployment.**

Copied from frozen production training (`production/hybrid/`).

## model.keras

| Property | Value |
|----------|-------|
| Purpose | Trained Hybrid GRU weights |
| Format | Keras SavedModel |
| Loaded by | `app/services/artifact_loader.py` via TensorFlow |
| Changes | Only when releasing `hybrid_v2` |

**Architecture:** GRU(256)→Dropout(0.2)→GRU(128)→Dropout(0.2)→GRU(64)→Dense(128,relu)→Dense(96)

**Input tensor:** `(batch, 96, 1)` — last 96 scaled residuals  
**Output tensor:** `(batch, 96)` — predicted scaled residuals

## scalers.pkl

| Property | Value |
|----------|-------|
| Purpose | Per-container MinMaxScaler (435 known containers) |
| Format | Python pickle dict `{container_id: MinMaxScaler}` |
| Used when | `container_id` matches a training container |
| Changes | Only with new production training |

## residual_stats.pkl

| Property | Value |
|----------|-------|
| Purpose | Global residual z-score constants |
| Keys | `res_mean`, `res_std`, `input_window`, `forecast_horizon` |
| Used by | Every request (never refit) |
| Example | `res_std ≈ 0.1067` |

## production_config.json

| Property | Value |
|----------|-------|
| Purpose | Frozen model metadata snapshot |
| Contains | Hyperparameters, Prophet flags, model type |
| Used by | `/model/info` endpoint |

## prophet_metadata.pkl

| Property | Value |
|----------|-------|
| Purpose | Prophet configuration snapshot from training |
| Note | Prophet is **refit per request** on client history; this file documents training-time defaults |

## train_container_ids.json

| Property | Value |
|----------|-------|
| Purpose | List of 435 container IDs with frozen scalers |
| Used by | `/model/info` known count; scaler lookup |

## Upgrading artifacts

1. Train new production model (separate process)
2. Copy new bundle to `artifacts/hybrid_v2/`
3. Set `FORECAST_ARTIFACTS_DIR` and `FORECAST_MODEL_VERSION`
4. Restart service
5. Run integration tests

Never hot-swap files while service is running.
