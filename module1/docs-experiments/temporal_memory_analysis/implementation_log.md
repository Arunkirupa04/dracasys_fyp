# Implementation Log

## 2026-07-26 — TMA v1.0 initial implementation

### Module structure

| Path | Role |
|------|------|
| `utils/tma/config.py` | Frozen constants |
| `utils/tma/data.py` | Read-only series loading |
| `utils/tma/acf_pacf.py` | Long-lag ACF/PACF |
| `utils/tma/periodicity.py` | Daily lag analysis + bootstrap |
| `utils/tma/fft_analysis.py` | FFT/PSD summaries |
| `utils/tma/memory_length.py` | Effective memory metrics |
| `utils/tma/window_sufficiency.py` | Cumulative capture + hypothetical windows |
| `utils/tma/statistics.py` | Paired day1 vs beyond tests |
| `utils/tma/plots.py` | Publication figures (PNG/PDF) |
| `utils/tma/report.py` | Final report generation |

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_temporal_memory_analysis.py` | End-to-end experiment runner |
| `scripts/build_temporal_memory_analysis_notebook.py` | Notebook builder |

### Authoritative run

```
experiments/temporal_memory_analysis_2026-07-26_170052/
```

Runtime ~48 s on local environment (99 containers, 672-lag ACF).

### Design decisions

1. **Full train+val timeline** for CPU and Prophet-equivalent residuals to enable 672-lag analysis.
2. **Fourier OLS** instead of Prophet.fit() for read-only Prophet-equivalent residuals.
3. **Frozen baseline inference cache** for hybrid day-1 post-forecast residuals (96 steps).
4. **Matplotlib Agg backend** and subsampled heatmap lags for stable batch plotting.

### Outputs generated

- 10 plot categories × 3 series types + comparison figures + 3 case studies
- JSON/CSV tables under `statistics/`, `acf/`, `pacf/`, `fft/`, `tables/`
- `reports/final_report.md` and `reports/final_report.json`

### 2026-07-26 — Documentation interpretation update (no re-run)

Expanded interpretation across `docs/temporal_memory_analysis/` and `reports/final_report.md`:

- Three distinct signals (CPU, Prophet residual, Hybrid post-forecast) — never mix
- Architecture-specific conclusions (Global GRU vs Hybrid GRU)
- Relationship to RPA, CSRLE, LFHE, RRA Audit
- Explicit "what TMA does NOT claim" subsection
- Thesis-ready wording in `summary.md` and `discussion.md`

**Numerical results unchanged.** No analysis re-run, no artifact modification.
