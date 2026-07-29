# Discussion

## Expected interpretation framework

### If V1 improves most

Prophet level context helps the GRU condition corrections on operating point — consistent with context feasibility study (level-regime coupling).

### If V2 improves most

Explicit volatility channel addresses heteroscedasticity / dispersion — aligns with LFHE finding that shape and scale decouple.

### If V4 does not beat V2/V3

Peak flag may be redundant with Prophet level; prefer parsimonious model.

### If no variant improves materially

Real Alibaba residual signal remains weak (RLLA r≈0.02–0.06); context alone insufficient — loss function or architecture changes needed.

## Relation to prior work

| Study | HCERL connection |
|-------|------------------|
| Context feasibility | Feature set justification |
| TMA | 96-step window unchanged; calendar excluded |
| LFHE | Monitor std ratio alongside MAE |
| RLLA | Bounds expected effect size |
| CSRLE | GRU capacity not the bottleneck on synthetic data |
