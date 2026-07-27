# Glossary

| Term | Definition |
|------|------------|
| **Hybrid model** | Prophet seasonal forecast + GRU residual forecast combined |
| **Prophet** | Facebook Prophet time-series model for seasonality |
| **GRU** | Gated Recurrent Unit neural network |
| **Residual** | `CPU_scaled - Prophet_prediction` — signal the GRU learns |
| **Input window** | 96 past residual steps fed to GRU |
| **Horizon** | Number of future steps predicted (max 96) |
| **Day-1 forecast** | First 96 steps = 24 hours at 15-min sampling |
| **cpu_scaled** | CPU normalised to [0,1] via MinMaxScaler |
| **residual_scaled** | Global z-scored residual using training mean/std |
| **Known container** | One of 435 training containers with frozen scaler |
| **Scaler mode** | `known` (frozen scaler) or `new` (fit on request) |
| **Artifact bundle** | Directory with model.keras, scalers.pkl, etc. |
| **hybrid_v1** | Current frozen model version identifier |
