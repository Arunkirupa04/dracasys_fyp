# Peak Definition Decision — Phase 2 Task 7

**Decision date:** 2026-07-14  
**Status:** LOCKED for remainder of Peak-Aware Hybrid research

---

## Chosen Definition

A CPU utilization timestep is labelled a **peak** if:

```
cpu_real(container, t) >= P90(container_train)
```

Where:

- `cpu_real` = per-container inverse MinMax transform of `cpu_scaled` (real CPU %)
- `P90(container_train)` = 90th percentile of that container's **train-period only** real CPU values
- Thresholds are **never** refit on validation or test data

**Chosen percentile:** **P90**

---

## Rejected Alternatives

| Alternative | Reason for rejection |
|-------------|---------------------|
| **P85** | Too permissive (~22% train peaks); limited additional discrimination vs P90 |
| **P95** | Too conservative (~11% train peaks); may weaken peak-aware training signal |
| **Global threshold** | Invalid under per-container MinMax scaling |
| **Absolute CPU floor** | Not required; affects only 5/99 near-idle containers; documented as limitation |
| **Sequence-level weighting** | >92% of sequences are peak sequences (Task 3) |

---

## Phase 3 Implications

| Decision | Choice |
|----------|--------|
| Weighting granularity | **Timestep-weighted MSE** |
| Early stopping | **Unweighted val_loss** (unchanged from baseline) |
| Peak weight λ | To be justified in Phase 3 (suggested starting point: 5) |

---

## Limitations (document in thesis)

1. Five near-idle containers with zero thresholds inflate peak statistics
2. Validation peak rates exceed train rates under fixed thresholds
3. Spiky workload stratification partially artefact-driven

---

*This decision is recorded in `peak_definition_decision.json` and `docs/peak-aware-hybrid/phase-02-peak-exploration.md`.*
