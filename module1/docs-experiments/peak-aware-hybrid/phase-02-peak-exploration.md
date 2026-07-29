# Phase 2 — Peak Exploration

[← Overview](README.md)

---

## 3. Phase 2 — Peak Exploration

### 3.1 Objective

Analyze peak behaviour in the training data and select **one evidence-based peak definition** before any Peak-Aware implementation. This phase is exploratory only: no model training, no baseline modification.

### 3.2 Methodology (Phase 2)

Peak labelling rules applied across Tasks 2–7:

| Rule | Specification |
|------|---------------|
| Value space | Real CPU utilization percent |
| Derivation | Per-container inverse MinMax transform of `cpu_scaled` |
| Threshold scope | Per-container |
| Threshold fit period | Train period only |
| Candidate percentiles | P85, P90, P95 |
| Peak timestep | `cpu_real >= threshold(container, P)` |
| Sequence configuration | `input_window=96`, `forecast_horizon=96` (matches baseline) |
| Validation use | Feasibility counts only; thresholds are **not** tuned on validation data |

### 3.3 Assumptions

- The frozen preprocessing artifacts in `data/` accurately represent the training and validation periods.
- Per-container percentile thresholds on real CPU % are interpretable across heterogeneous workloads.
- The 99-container evaluation cohort is representative for peak characterization.
- Workload labels in `container_metadata.parquet` (`pattern_type`: stable / medium / spiky) are valid for stratification.

### 3.4 Cohort

| Item | Value |
|------|-------|
| Selected containers | 100 |
| Present in frozen train data | 99 |
| Excluded | `c_14674` |
| Workload distribution (99 containers) | stable: 35, medium: 33, spiky: 31 |

### 3.5 Experiment Location

```
experiments/peak_exploration_2026-07-14/
```

---

### Task 1 — Prepare the Peak Exploration Workspace

**Status:** Complete  
**Date:** 2026-07-14

#### Objective

Create the experiment folder structure, initialize placeholder output files for all Phase 2 tasks, and establish this research document as the primary research log. No data analysis was performed in this task.

#### Methodology

1. Created a timestamped experiment directory under `experiments/`.
2. Defined output CSV schemas (headers only) mapped to Tasks 2–7.
3. Created a `figures/` directory for Phase 2 visualizations.
4. Wrote `experiment_metadata.json` recording experiment scope, data sources, peak-labelling rules, and task status.
5. Initialized this research document with Phase 1 summary and Phase 2 framework.

#### Implementation Decisions

| Decision | Rationale |
|----------|-----------|
| Single research log (`docs/peak_aware_hybrid_research.md`) | Avoids fragmented documentation; supports thesis reuse |
| Flat CSV layout in experiment folder | Simple, discoverable outputs aligned with the Phase 2 plan |
| Separate `timestep_summary.csv` and `sequence_peak_stats.csv` | Distinguishes Task 2 and Task 3 deliverables clearly |
| `experiment_metadata.json` for machine-readable provenance | Supports reproducibility without duplicating narrative in the research log |
| No analysis scripts in Task 1 | Workspace preparation only; analysis begins in Task 2 after approval |

#### Generated Outputs

```
experiments/peak_exploration_2026-07-14/
├── experiment_metadata.json
├── per_container_peak_stats.csv      # Task 2 — headers only
├── timestep_summary.csv              # Task 2 — headers only
├── sequence_peak_stats.csv           # Task 3 — headers only
├── workload_stratification.csv       # Task 4 — headers only
├── val_sanity_check.csv              # Task 5 — headers only
├── threshold_comparison.csv          # Task 6 — headers only
└── figures/
    └── .gitkeep
```

| Output file | Populated by task | Purpose |
|-------------|-------------------|---------|
| `per_container_peak_stats.csv` | Task 2 | Per-container thresholds and peak timestep rates |
| `timestep_summary.csv` | Task 2 | Global timestep peak statistics |
| `sequence_peak_stats.csv` | Task 3 | Sliding-window sequence peak statistics |
| `workload_stratification.csv` | Task 4 | Peak rates by workload type |
| `val_sanity_check.csv` | Task 5 | Validation feasibility using train thresholds |
| `threshold_comparison.csv` | Task 6 | Consolidated P85 / P90 / P95 comparison |
| `figures/` | Tasks 2–6 | Exploration visualizations |

#### Findings

No analytical findings in Task 1. The workspace and documentation framework are ready.

#### Conclusions

Task 1 is complete. The Phase 2 experiment directory is initialized, output schemas are defined, and the research log is established. Analysis has not started.

#### Limitations

- Output files contain headers only; they must be populated in subsequent tasks.
- No exploration script or notebook was created in Task 1; implementation of analysis will occur task-by-task after approval.

#### Next Task

**Task 2 — Peak Timestep Analysis:** Compute peak timestep statistics for P85, P90, and P95 on the frozen training dataset; populate `per_container_peak_stats.csv` and `timestep_summary.csv`; generate tables and figures; update this document.

---

### Task 2 — Peak Timestep Analysis

**Status:** Complete  
**Date:** 2026-07-14

#### Objective

Quantify how frequently CPU utilization peaks occur at the timestep level in the frozen training dataset, comparing per-container percentile thresholds P85, P90, and P95 on real CPU utilization percent.

#### Methodology

1. Loaded frozen artifacts: `data/train_df.parquet`, `data/scalers.pkl`, `data/selected_containers.npy`, and `data/container_metadata.parquet`.
2. Restricted analysis to **99 evaluable containers** present in the train split (`c_14674` excluded).
3. Converted each container's `cpu_scaled` values to **real CPU %** using the per-container `MinMaxScaler` inverse transform.
4. For each container, computed train-only thresholds at P85, P90, and P95.
5. Labelled peak timesteps where `cpu_real >= threshold`.
6. Aggregated per-container and global statistics.
7. Saved outputs and generated publication-quality figures (PNG and PDF).

**Reproducibility script:** `scripts/peak_exploration_task2_timesteps.py`

#### Assumptions

- Peak characterization on the train period is sufficient for threshold selection (validation reserved for Task 5).
- Inverse-transformed `cpu_scaled` is the correct real CPU % representation, consistent with baseline evaluation.
- Per-container percentile thresholds adapt to heterogeneous workload baselines.

#### Implementation Decisions

| Decision | Rationale |
|----------|-----------|
| New standalone script under `scripts/` | Keeps baseline modules untouched; supports reproducibility |
| Real CPU % via inverse MinMax | Avoids misleading thresholds on per-container scaled values |
| `>=` percentile threshold | Standard upper-tail peak definition; ~5/10/15% nominal rates for P95/P90/P85 |
| Three figures for Task 2 | Global comparison, distribution, and top-container inspection |

#### Generated Outputs

| Output | Location |
|--------|----------|
| Per-container statistics | `experiments/peak_exploration_2026-07-14/per_container_peak_stats.csv` |
| Global timestep summary | `experiments/peak_exploration_2026-07-14/timestep_summary.csv` |
| Machine-readable summary | `experiments/peak_exploration_2026-07-14/task2_results.json` |
| Global peak rate chart | `figures/task2_global_peak_timestep_rate.png` / `.pdf` |
| Per-container distribution | `figures/task2_per_container_peak_distribution.png` / `.pdf` |
| Top containers (P90) | `figures/task2_top_containers_p90.png` / `.pdf` |

#### Results — Global Peak Timestep Summary (Train Period)

| Percentile | Global peak timestep % | Mean per-container % | Median per-container % | Min per-container % | Max per-container % |
|------------|------------------------|----------------------|------------------------|---------------------|---------------------|
| **P85** | 21.66% | 21.67% | 16.29% | 14.98% | 100.0% |
| **P90** | 17.36% | 17.37% | 11.58% | 10.00% | 100.0% |
| **P95** | 11.05% | 11.06% | 5.73% | 5.00% | 100.0% |

- **Containers analyzed:** 99  
- **Total train timesteps:** 60,658  
- **Containers with zero peaks:** 0 (under all three percentiles)

#### Findings

1. **Peak timesteps are not extremely rare** under per-container percentile thresholds. Even P95 yields ~11% global peak timesteps — well above a near-zero rate that would make peak-aware learning ineffective.

2. **P90 and P95 fall within the pre-specified useful range** (approximately 5–17% globally). P85 is higher (~22%) and labels more timesteps as peaks, which may overweight normal variation.

3. **Wide cross-container variation** exists (std ≈ 18–20 percentage points on per-container rates). Some containers show much higher peak rates than others.

4. **Five containers show 100% peak timesteps at P90** (`c_13308`, `c_14106`, `c_15035`, `c_15446`, `c_15794`). Investigation shows these are near-idle workloads where `threshold_p90 = 0.0` after inverse transform (mean real CPU &lt; 0.06%). When the threshold is zero, every timestep trivially satisfies `cpu_real >= 0`.

5. **Minimum per-container peak rates** align with theoretical tail fractions: ~15% (P85), ~10% (P90), ~5% (P95) for typical containers without degenerate thresholds.

#### Unexpected Observations

The **zero-threshold artefact** on near-idle containers is the main unexpected result. A supplementary impact analysis (below) was conducted to quantify its effect on global peak statistics. No methodological changes were made.

#### Conclusions

- **P95 (~11%)** and **P90 (~17%)** are both viable candidate definitions from a frequency perspective.
- **P85 (~22%)** is looser and may be too permissive for a "peak" label in thesis language.
- **P90** remains the leading candidate, pending sequence analysis (Task 3), workload stratification (Task 4), and validation feasibility (Task 5).
- The near-idle container issue disproportionately inflates global peak counts and should be discussed in the final peak definition decision (Task 7).

#### Limitations

- Analysis covers train timesteps only; validation feasibility is not yet assessed.
- 15-minute resampling and interpolation in preprocessing may smooth very short spikes.
- Short train history (~614 steps per container) makes percentile estimates coarse for low-activity containers.
- No absolute CPU floor was applied in this task (by design).

---

### Task 2 Supplement — Near-Idle Zero-Threshold Impact Analysis

**Status:** Complete  
**Date:** 2026-07-14

#### Objective

Quantify the impact of near-idle containers whose per-container P90 threshold equals 0.0, causing all timesteps to be labeled as peaks. This analysis is exploratory only: it does **not** modify the dataset, peak definition, or Phase 2 methodology.

#### Methodology

1. Identified containers with `threshold_p90 == 0.0` from Task 2 outputs.
2. Summarized their train timesteps, CPU characteristics, and peak timestep contributions at P85, P90, and P95.
3. Computed **adjusted global peak percentages** excluding these containers (analysis-only exclusion, not applied to frozen data).
4. Compared original vs adjusted global rates to assess whether the artefact is localized or materially skews overall statistics.

**Reproducibility script:** `scripts/peak_exploration_near_idle_impact.py`

#### Assumptions

- Task 2 per-container statistics are correct and sufficient for this derivative analysis.
- Exclusion of the five containers is a counterfactual for interpretation only; the frozen dataset remains unchanged.
- Peak timestep counts are derived as `round(train_timesteps × peak_timestep_pct / 100)`.

#### Affected Containers (P90 threshold = 0.0)

| Container ID | Pattern | Train steps | Mean CPU % | Max CPU % | P85 thr | P90 thr | P95 thr |
|--------------|---------|-------------|------------|-----------|---------|---------|---------|
| c_13308 | spiky | 614 | 0.0004 | 0.25 | 0.0 | 0.0 | 0.0 |
| c_14106 | spiky | 608 | 0.0005 | 0.33 | 0.0 | 0.0 | 0.0 |
| c_15035 | spiky | 609 | 0.0006 | 0.25 | 0.0 | 0.0 | 0.0 |
| c_15446 | spiky | 614 | 0.056 | 5.00 | 0.0 | 0.0 | 0.25 |
| c_15794 | spiky | 612 | 0.0038 | 2.33 | 0.0 | 0.0 | 0.0 |

All five are labeled **spiky** in `container_metadata.parquet` (high coefficient of variation from near-zero means), despite being near-idle in absolute CPU terms.

#### Dataset Share

| Metric | Value |
|--------|-------|
| Affected containers | 5 of 99 (5.1%) |
| Affected train timesteps | 3,057 of 60,658 |
| Share of total train timesteps | **5.04%** |

#### Peak Timesteps Contributed by Affected Containers

| Percentile | Peak timesteps contributed | % of all peak timesteps (dataset-wide) |
|------------|---------------------------|----------------------------------------|
| **P85** | 3,057 | 23.3% |
| **P90** | 3,057 | 29.0% |
| **P95** | 2,476 | 36.9% |

At P85/P90, every timestep in these containers is labeled a peak (3,057 peaks each). At P95, `c_15446` contributes only 33 peak timesteps (threshold 0.25), reducing the idle contribution slightly.

#### Adjusted Global Peak Rates (Excluding 5 Containers, Analysis Only)

| Percentile | Original global peak % | Adjusted global peak % | Change (pp) |
|------------|------------------------|------------------------|-------------|
| **P85** | 21.66% | 17.50% | −4.16 |
| **P90** | 17.36% | 12.98% | −4.39 |
| **P95** | 11.05% | 7.34% | −3.71 |

#### Generated Outputs

| Output | Location |
|--------|----------|
| Affected container details | `experiments/peak_exploration_2026-07-14/near_idle_zero_threshold_containers.csv` |
| Adjusted vs original summary | `experiments/peak_exploration_2026-07-14/near_idle_impact_summary.csv` |
| Machine-readable results | `experiments/peak_exploration_2026-07-14/near_idle_impact.json` |

#### Findings

1. The issue is **localized in terms of timesteps** — only **5.04%** of train data comes from these five containers.
2. The issue is **material in terms of peak labeling** — these containers contribute **23–37%** of all peak timesteps under P85–P95, because every (or nearly every) timestep is labeled a peak.
3. Excluding them for analysis reduces global peak rates by **3.7–4.4 percentage points**, showing they **meaningfully inflate** Task 2 global statistics.
4. All affected containers are classified as **spiky** due to high CV from near-zero means, which may distort workload-stratification interpretations in Task 4.

#### Conclusions

The zero-threshold behaviour is a **localized artefact affecting a small timestep fraction (5%)**, but it has a **disproportionate and meaningful influence on overall peak statistics (up to 37% of peak labels)**. It should be:

- **Documented** as a limitation in Phase 2 and the thesis.
- **Discussed** in Task 7 when finalizing the peak definition.
- **Monitored** in Tasks 3–6 (especially workload stratification).

No change to the peak definition or methodology is made at this stage, per the research plan.

#### Influence on Next Task

Task 3 sequence analysis should proceed with the **original percentile rules unchanged**. Results should be interpreted with awareness that ~5% of sequences may come from near-idle containers with degenerate thresholds. Task 4 spiky-container statistics may be partially driven by these containers.

#### Next Task

**Task 3 — Peak Sequence Analysis** (awaiting approval, original Phase 2 plan).

---

### Task 3 — Peak Sequence Analysis

**Status:** Complete  
**Date:** 2026-07-14

#### Objective

Quantify how CPU utilization peaks appear at the **sequence level** under the baseline 96→96 sliding window configuration, using the unchanged per-container percentile thresholds (P85, P90, P95) from Task 2. Determine whether peaks within 24-hour forecast horizons are sparse or frequent, and whether timestep-level or sequence-level weighting is more appropriate for Phase 3.

#### Methodology

1. Loaded frozen train data, scalers, and the 99-container evaluation cohort.
2. Applied per-container train-only percentile thresholds (same rules as Task 2).
3. Generated sliding windows matching the baseline Hybrid GRU: `input_window=96`, `forecast_horizon=96`.
4. For each sequence, counted peak timesteps in the **forecast horizon only** (not the input window).
5. Labelled a **peak sequence** when the horizon contains ≥ 1 peak timestep.
6. Computed summary statistics and distributions for P85, P90, and P95.
7. Produced a **reference comparison** excluding the five near-idle zero-threshold containers (`c_13308`, `c_14106`, `c_15035`, `c_15446`, `c_15794`) — analysis only, dataset unchanged.

**Reproducibility script:** `scripts/peak_exploration_task3_sequences.py`

#### Sequence Generation Process

The windowing logic mirrors `utils/sequence_utils.create_residual_sequences()`:

```
For each container (chronologically sorted):
    max_start = len(container) - input_window - forecast_horizon
    For start in 0 .. max_start - 1:
        horizon = cpu_real[start + input_window : start + input_window + forecast_horizon]
        peak_count = count(timesteps in horizon where cpu_real >= threshold)
        peak_sequence = (peak_count >= 1)
```

| Parameter | Value |
|-----------|-------|
| Input window | 96 timesteps (context; not peak-labelled) |
| Forecast horizon | 96 timesteps (peak-labelled region) |
| Total sequences (official) | 41,650 |
| Total sequences (reference, 94 containers) | 39,553 |

#### Official Results — All 99 Containers

| Percentile | Total sequences | Peak sequences | Peak sequence % | Mean peak steps / seq | Median peak steps / seq | Zero-peak seq | Full-peak seq (96/96) |
|------------|-----------------|----------------|-----------------|----------------------|------------------------|---------------|----------------------|
| **P85** | 41,650 | 40,247 | **96.63%** | 20.32 | 17.0 | 1,403 (3.37%) | 2,097 (5.03%) |
| **P90** | 41,650 | 40,065 | **96.19%** | 16.26 | 12.0 | 1,585 (3.81%) | 2,097 (5.03%) |
| **P95** | 41,650 | 38,429 | **92.27%** | 10.55 | 6.0 | 3,221 (7.73%) | 1,675 (4.02%) |

#### Reference Comparison — Excluding 5 Near-Idle Containers (Analysis Only)

| Percentile | Total sequences | Peak sequences | Peak sequence % | Mean peak steps / seq | Median peak steps / seq | Zero-peak seq | Full-peak seq (96/96) |
|------------|-----------------|----------------|-----------------|----------------------|------------------------|---------------|----------------------|
| **P85** | 39,553 | 38,150 | **96.45%** | 16.31 | 16.0 | 1,403 (3.55%) | 0 (0.00%) |
| **P90** | 39,553 | 37,968 | **95.99%** | 12.03 | 11.0 | 1,585 (4.01%) | 0 (0.00%) |
| **P95** | 39,553 | 36,332 | **91.86%** | 6.98 | 6.0 | 3,221 (8.14%) | 0 (0.00%) |

**Reference vs official deltas (P90 example):**

| Metric | Official | Reference | Change |
|--------|----------|-----------|--------|
| Peak sequence % | 96.19% | 95.99% | −0.20 pp |
| Mean peak steps / seq | 16.26 | 12.03 | −4.23 steps |
| Full-peak sequences (96/96) | 2,097 | 0 | −2,097 |

#### Distribution of Peak Timesteps Within Sequences (P90)

Peaks are **spread across many levels** within horizons rather than concentrated at a single count. The most common bins (official cohort) centre around **7–14 peak timesteps** per 96-step horizon (~5–6% of sequences each). At P90:

- **1,585 sequences (3.8%)** contain zero peak timesteps.
- **2,097 sequences (5.0%)** contain all 96 timesteps as peaks (entirely from near-idle containers).
- The remaining ~91% of sequences contain a **partial** number of peak timesteps.

After excluding near-idle containers, the distribution is similar in shape but **no sequences have all 96 peak timesteps**, and mean peak steps per sequence drops by ~26%.

#### Generated Outputs

| Output | Location |
|--------|----------|
| Official sequence summary | `sequence_peak_stats.csv` |
| Reference sequence summary | `sequence_peak_stats_reference.csv` |
| Peak-step distribution | `sequence_peak_distribution.csv` |
| Per-container sequence stats | `per_container_sequence_stats.csv` |
| Machine-readable results | `task3_results.json` |
| Peak sequence rate comparison | `figures/task3_peak_sequence_rate.png` / `.pdf` |
| Mean peak steps comparison | `figures/task3_mean_peak_timesteps_per_sequence.png` / `.pdf` |
| Distribution plots (P85/P90/P95, both cohorts) | `figures/task3_peak_step_distribution_p*_official/reference.png` / `.pdf` |

#### Interpretation

**What the statistics indicate**

1. **Peak sequences are extremely common** — over **92–97%** of all training sequences contain at least one peak timestep in the forecast horizon, regardless of percentile.
2. **Peaks are not sparse within horizons** — when peaks are present, they typically span **6–17 timesteps** on average (median 6–17 depending on percentile), not just 1–2 isolated steps.
3. **Sequence-level and timestep-level weighting behave very differently:**
   - **Sequence-level weighting** would up-weight **~92–97% of all sequences**, leaving only **3–8%** unweighted. It is therefore a **blunt instrument** — nearly equivalent to weighting the entire dataset with a minor adjustment.
   - **Timestep-level weighting** can target the **specific peak timesteps** within each horizon (on average ~10–20 of 96 steps at P85–P90), leaving non-peak timesteps at weight 1.0. This provides **finer control** and a more meaningful distinction.

**Official vs reference comparison**

- Near-idle containers **do not materially change peak sequence %** (−0.2 pp at P90) because they contribute sequences that are **always peak sequences** — they increase the numerator and denominator proportionally.
- They **do materially inflate mean peak timesteps per sequence** (+4.2 steps at P90) and create **all-96-peak sequences** (2,097 sequences, 5% of total) that disappear in the reference cohort.
- Zero-peak sequence counts are **identical** in official and reference (1,585 at P90) — near-idle containers contribute **no** zero-peak sequences.

#### Observations

1. The **dominant pattern** is mixed: most sequences contain a moderate number of peak timesteps, not zero and not all 96.
2. **P95** is the only percentile where a meaningful fraction of sequences lack peaks (~7.7% zero-peak).
3. Near-idle containers inflate **intensity** (peak steps per sequence) more than **prevalence** (peak sequence %).
4. The 2,097 full-peak sequences match the near-idle container sequence count (~5 × ~419 windows), confirming the Task 2 supplement finding at sequence level.

#### Limitations

- Peak labels apply to the forecast horizon only; the 96-step input context is not peak-labelled (consistent with GRU training targets).
- Analysis is train-period only; validation sequence feasibility is assessed in Task 5.
- Per-container percentile thresholds inherit the near-idle zero-threshold artefact documented in Task 2 supplement.
- Sequence counts assume the same windowing logic as `create_residual_sequences`; off-by-one differences would not materially change conclusions.

#### Impact on Phase 3 (Peak-Aware Learning Design)

| Recommendation | Evidence |
|----------------|----------|
| **Prefer timestep-weighted MSE** | Only ~3–8% of sequences have zero peaks; sequence weighting cannot distinguish the majority of training data |
| **Use P90 or P95 as leading threshold** | P95 retains more zero-peak sequences (7.7%) if sequence-level weighting is considered; P90 remains the balanced candidate from Task 2 |
| **Document near-idle limitation** | Discuss in Task 7; mean peak-step intensity is sensitive to the five containers |
| **Do not rely on sequence-level weighting alone** | Would up-weight >92% of sequences, providing minimal selective emphasis |

#### Next Task

**Task 4 — Workload Stratification** (awaiting approval).

---

### Task 4 — Workload Stratification

**Status:** Complete  
**Date:** 2026-07-14

#### Objective

Compare peak behaviour across workload types (**stable**, **medium**, **spiky**) using `pattern_type` from `container_metadata.parquet`, at both timestep and sequence levels, for P85, P90, and P95. Assess whether peaks are concentrated in spiky containers as hypothesized, and quantify the distorting effect of near-idle containers.

#### Methodology

1. Joined Task 2 per-container timestep statistics and Task 3 per-container sequence statistics with `pattern_type` labels.
2. Aggregated **weighted** metrics per workload group (timesteps weighted by `train_timesteps`; sequences weighted by `total_sequences`).
3. Produced **official** results (99 containers: 35 stable, 33 medium, 31 spiky) and a **reference comparison** excluding five near-idle spiky containers (26 spiky remaining).
4. Generated comparison tables and workload-stratified visualizations.

**Reproducibility script:** `scripts/peak_exploration_task4_workload.py`

#### Assumptions

- `pattern_type` from preprocessing EDA (CV tertiles) is the correct workload stratification label.
- Weighted aggregation by timestep/sequence count reflects dataset contribution more accurately than unweighted container means.
- Near-idle reference comparison is interpretive only; methodology and dataset remain unchanged.

#### Official Results — Timestep Level (Weighted Peak Timestep %)

| Percentile | Stable (n=35) | Medium (n=33) | Spiky (n=31) |
|------------|---------------|---------------|--------------|
| **P85** | 17.02% | 17.66% | **31.17%** |
| **P90** | 12.63% | 12.03% | **28.38%** |
| **P95** | 6.94% | 7.14% | **19.86%** |

#### Official Results — Sequence Level (Weighted)

| Percentile | Stable peak seq % | Medium peak seq % | Spiky peak seq % | Spiky mean peak steps |
|------------|-------------------|-------------------|------------------|----------------------|
| **P85** | 98.32% | 97.17% | 94.16% | 29.37 |
| **P90** | 97.51% | 96.77% | 94.09% | 26.73 |
| **P95** | 93.00% | 93.20% | 90.44% | 19.13 |

#### Reference Comparison — Timestep Level (Excl. 5 Near-Idle Spiky)

| Percentile | Stable | Medium | Spiky (n=26) |
|------------|--------|--------|--------------|
| **P85** | 17.02% | 17.66% | **17.96%** |
| **P90** | 12.63% | 12.03% | **14.64%** |
| **P95** | 6.94% | 7.14% | **8.13%** |

#### Reference Comparison — Sequence Level (Excl. Near-Idle)

| Percentile | Stable peak seq % | Medium peak seq % | Spiky peak seq % | Spiky mean peak steps |
|------------|-------------------|-------------------|------------------|----------------------|
| **P85** | 98.32% | 97.17% | 93.03% | 16.60 |
| **P90** | 97.51% | 96.77% | 92.96% | 13.45 |
| **P95** | 93.00% | 93.20% | 88.61% | 7.87 |

After excluding near-idle containers, spiky-group mean peak steps at P90 drop from **26.73 → 13.45**, aligning closely with stable (**11.97**) and medium (**10.97**).

#### Generated Outputs

| Output | Location |
|--------|----------|
| Workload stratification table | `workload_stratification.csv` |
| Machine-readable results | `task4_results.json` |
| Timestep rate by workload (official / reference) | `figures/task4_peak_timestep_by_workload_*.png` / `.pdf` |
| Sequence rate and mean steps by workload | `figures/task4_peak_sequence_by_workload_*.png` / `.pdf` |
| CV vs peak timestep scatter (P90) | `figures/task4_cv_vs_peak_timestep_p90.png` / `.pdf` |

#### Observations

1. **Official spiky timestep rates appear much higher** (28–31% at P85–P90) than stable/medium (~12–18%). This largely reflects the **five near-idle spiky containers** (5 of 31 spiky = 16.1% of the spiky group).
2. **After near-idle exclusion**, spiky timestep rates **converge with stable and medium** groups. Residual spiky elevation at P90 (+1.4 pp) and P95 (+1.2 pp) is modest.
3. **Peak sequence rates are high across all workload types** (88–98%), confirming Task 3 — workload type does not strongly separate peak vs non-peak sequences.
4. **Mean peak steps per sequence** is the metric most distorted by near-idle spiky containers (official spiky ~27 vs reference ~13 at P90).
5. The **CV vs peak scatter plot** shows near-idle spiky containers as extreme outliers (very high CV, 100% peak rate), separate from the main cloud of stable/medium/spiky containers.

#### Interpretation

The hypothesis that peaks are concentrated in **spiky containers** is **partially supported only before accounting for near-idle artefacts**. Under the unchanged percentile methodology:

- **Timestep level:** True spiky workloads show only marginally higher peak rates than stable/medium containers.
- **Sequence level:** All workload types exhibit very high peak-sequence prevalence; stratification is more useful for **Phase 5 evaluation reporting** than for training design.
- **Near-idle misclassification** as "spiky" (high CV from near-zero means) is the dominant driver of apparent spiky–stable separation in official statistics.

#### Limitations

- `pattern_type` is derived from CV on the full container series, not from peak frequency; spiky label ≠ high absolute CPU.
- Five near-idle containers disproportionately affect the spiky group (16% of spiky containers, but 26% of spiky train timesteps in reference vs 31 official).
- Small group sizes (26–35 containers per type) limit statistical precision; no formal tests were conducted (per research plan).

#### Impact on Phase 3 and Remaining Phase 2 Tasks

| Implication | Detail |
|-------------|--------|
| Phase 3 weighting | Timestep-level weighting remains preferred; workload type does not provide a simpler alternative |
| Task 5–7 | Report Phase 5 metrics by workload type, but interpret spiky-group results cautiously |
| Task 7 decision | Near-idle mislabelling should be discussed; optional absolute floor remains a Task 7 consideration, not introduced here |
| Peak-aware target | Model should not assume spiky containers are the sole peak source — peaks are widespread across workload types |

#### Next Task

**Task 5 — Validation Sanity Check** (awaiting approval).

---

### Task 5 — Validation Sanity Check

**Status:** Complete  
**Date:** 2026-07-14

#### Objective

Verify that sufficient peak timesteps and peak evaluation horizons exist in the **validation period** when train-fitted per-container thresholds (P85, P90, P95) are applied. Confirm that Phase 5 peak-only metrics will be measurable. Thresholds were **not** tuned or modified using validation data.

#### Methodology

1. Computed per-container thresholds on **train data only** (same rules as Tasks 2–4).
2. Applied thresholds to all validation timesteps for the 99-container cohort.
3. Compared train vs validation peak timestep rates.
4. Attempted 96→96 sliding-window sequence generation on validation data alone (same configuration as baseline).
5. Performed an **evaluation-aligned Day-1 horizon check**: first 96 validation timesteps per container, matching the baseline Hybrid Day-1 evaluation protocol.
6. Produced official and reference (excl. near-idle) comparisons.
7. Marked validation peak evaluation as **feasible** when `val_peak_timestep_pct ≥ 2%` and `val_day1_containers_with_peak_pct ≥ 50%` (pre-specified sanity criteria).

**Reproducibility script:** `scripts/peak_exploration_task5_validation.py`

#### Assumptions

- Validation labels use train-fitted thresholds only — no leakage from validation into threshold definition.
- The Day-1 horizon check (96 val steps per container) is the operationally relevant sequence unit for Phase 5 evaluation, because the validation period (~154 steps) is too short for 96→96 sliding windows on val data alone.
- Feasibility criteria are sanity checks, not threshold-selection rules.

#### Validation Period Characteristics

| Property | Value |
|----------|-------|
| Containers | 99 |
| Val timesteps per container | ~150–154 (mean 153.7) |
| Total val timesteps | 15,219 |
| Minimum for 96→96 sliding window | 192 timesteps |
| **96→96 sliding sequences on val only** | **0 (infeasible)** |

#### Official Results — Peak Timestep Rates (Train vs Validation)

| Percentile | Train peak timestep % | Val peak timestep % | Train/Val ratio |
|------------|----------------------|---------------------|-----------------|
| **P85** | 21.66% | **33.91%** | 1.57 |
| **P90** | 17.36% | **28.29%** | 1.63 |
| **P95** | 11.05% | **19.45%** | 1.76 |

Validation peak timestep rates are **higher** than train (ratio 1.4–1.8×), indicating peaks are present and somewhat more frequent in the evaluation period.

#### Official Results — Sequence-Level Checks

| Check | P85 | P90 | P95 |
|-------|-----|-----|-----|
| 96→96 sliding sequences on val | 0 | 0 | 0 |
| Val sliding peak sequence % | N/A | N/A | N/A |
| **Day-1 containers with ≥1 peak** | **98.99%** | **98.99%** | **97.98%** |
| **Day-1 peak timestep % (of 96 steps)** | 30.74% | **25.65%** | 17.47% |
| **Val peak evaluation feasible** | Yes | **Yes** | **Yes** |

#### Reference Comparison (Excl. 5 Near-Idle)

| Percentile | Val peak timestep % | Day-1 containers with peak % | Day-1 peak timestep % | Feasible |
|------------|---------------------|------------------------------|----------------------|----------|
| **P85** | 30.40% | 98.94% | 27.06% | Yes |
| **P90** | 24.48% | 98.94% | 21.70% | Yes |
| **P95** | 16.15% | 97.87% | 14.10% | Yes |

Near-idle exclusion modestly reduces validation peak rates but does not affect feasibility conclusions.

#### Generated Outputs

| Output | Location |
|--------|----------|
| Validation sanity summary | `val_sanity_check.csv` |
| Per-container Day-1 horizon detail | `val_day1_horizon_sanity.csv` |
| Machine-readable results | `task5_results.json` |
| Train vs val timestep comparison | `figures/task5_train_vs_val_peak_timestep_official.png` / `.pdf` |
| Train vs val (reference) | `figures/task5_train_vs_val_peak_timestep_reference.png` / `.pdf` |

#### Interpretation

1. **Peak timesteps are abundant in validation** — 16–34% of val timesteps are peaks depending on percentile. Phase 5 peak-only MAE/RMSE metrics will have sufficient support.
2. **96→96 sliding sequences cannot be built on validation data alone** — the val period (~154 steps) is shorter than `input_window + forecast_horizon` (192). This is expected and does not affect baseline evaluation, which uses train-context input and a single Day-1 forecast per container.
3. **Day-1 horizon check confirms evaluation feasibility** — 98–99% of containers have at least one peak timestep in their 96-step validation forecast horizon; Day-1 peak timestep rates (17–31%) support meaningful peak-subset evaluation.
4. **Validation peak rates exceed train rates** — no evidence that the evaluation period is peak-starved; if anything, peaks are more prevalent in validation than training under the fixed thresholds.
5. **Near-idle containers slightly inflate validation peak rates** but feasibility is unchanged after exclusion.

#### Observations

- The higher validation peak rate may reflect distribution shift between train/val chronological splits, or different workload behaviour in the held-out period — to be noted as a limitation, not used to retune thresholds.
- Only **1 container** (at P95) lacks peaks in the Day-1 horizon under official results — peak evaluation remains well-supported.

#### Limitations

- Validation sanity uses train-fitted thresholds; if val CPU distribution diverges strongly, label semantics may differ between periods.
- Sliding-window sequence checks on val are structurally infeasible; Day-1 horizon is the correct evaluation unit for this project.
- Feasibility thresholds (2% / 50%) are sanity guidelines, not statistical tests.

#### Impact on Phase 3 and Task 6–7

| Implication | Detail |
|-------------|--------|
| Phase 5 peak metrics | Feasible and well-supported at P85, P90, and P95 |
| Phase 5 evaluation unit | Use Day-1 96-step validation horizon per container (matches baseline) |
| Task 6 threshold comparison | Can proceed; validation feasibility confirmed for all candidates |
| Task 7 | Validation evidence supports selecting P90 or P95 without feasibility concerns |

#### Next Task

**Task 6 — Threshold Comparison** (awaiting approval).

---

### Task 6 — Threshold Comparison

**Status:** Complete  
**Date:** 2026-07-14

#### Objective

Consolidate evidence from Tasks 2–5 into a single comparison of P85, P90, and P95 per-container percentile thresholds. Summarize timestep behaviour, sequence behaviour, workload patterns, validation feasibility, and near-idle sensitivity. Discuss advantages and disadvantages to inform the Task 7 peak definition decision.

#### Methodology

1. Aggregated outputs from `timestep_summary.csv`, `sequence_peak_stats.csv`, `sequence_peak_stats_reference.csv`, `workload_stratification.csv`, `val_sanity_check.csv`, and `near_idle_impact_summary.csv`.
2. Built a unified `threshold_comparison.csv` table (one row per percentile).
3. Documented advantages and disadvantages for each threshold.
4. Generated a four-panel comparison figure.

**Reproducibility script:** `scripts/peak_exploration_task6_threshold_comparison.py`

No thresholds, datasets, or methodologies were modified.

#### Final Threshold Comparison Table (Official Cohort)

| Percentile | Train peak timestep % | Train peak sequence % | Zero-peak sequence % | Mean peak steps / seq | Val peak timestep % | Day-1 containers with peak % | Val feasible |
|------------|----------------------|----------------------|---------------------|----------------------|---------------------|---------------------------|--------------|
| **P85** | 21.66% | 96.63% | 3.37% | 20.32 | 33.91% | 98.99% | Yes |
| **P90** | 17.36% | 96.19% | 3.81% | 16.26 | 28.29% | 98.99% | Yes |
| **P95** | 11.05% | 92.27% | 7.73% | 10.55 | 19.45% | 97.98% | Yes |

#### Workload Behaviour (Weighted Train Peak Timestep %)

| Percentile | Stable | Medium | Spiky |
|------------|--------|--------|-------|
| **P85** | 17.02% | 17.66% | 31.17% |
| **P90** | 12.63% | 12.03% | 28.38% |
| **P95** | 6.94% | 7.14% | 19.86% |

Spiky elevation is largely near-idle-driven (Task 4 reference: spiky P90 falls to 14.64% after exclusion).

#### Near-Idle Sensitivity (Train Timestep %)

| Percentile | Official | Excl. near-idle | Change | Near-idle share of peak labels |
|------------|----------|-----------------|--------|-------------------------------|
| **P85** | 21.66% | 17.50% | −4.16 pp | 23.3% |
| **P90** | 17.36% | 12.98% | −4.39 pp | 29.0% |
| **P95** | 11.05% | 7.34% | −3.71 pp | 36.9% |

#### Advantages and Disadvantages

| Threshold | Advantages | Disadvantages |
|-----------|------------|---------------|
| **P85** | Highest peak coverage; strongest training signal; validation feasible | Too permissive (~22% peaks); poor zero-peak sequence discrimination; weak "spike" semantics |
| **P90** | Balanced frequency (~17% train / ~28% val); standard percentile; better zero-peak discrimination than P85; strong validation support; leading candidate across Tasks 2–5 | Peak-sequence rate still ~96% (sequence weighting ineffective); moderate near-idle distortion |
| **P95** | Strictest definition; best zero-peak sequence discrimination (7.7%); lowest near-idle label share; still validation-feasible | Fewest peak timesteps (~11% train); may miss moderate surges; weaker peak-aware training emphasis |

#### Generated Outputs

| Output | Location |
|--------|----------|
| Consolidated comparison table | `threshold_comparison.csv` |
| Advantages / disadvantages | `task6_results.json` |
| Four-panel comparison figure | `figures/task6_threshold_comparison.png` / `.pdf` |

#### Interpretation

1. **All three thresholds are validation-feasible** — Day-1 peak evaluation is well-supported for P85, P90, and P95.
2. **P85 is too loose** for a meaningful "peak" label in CPU forecasting research — over one-fifth of all timesteps are peaks, and sequence-level discrimination is minimal.
3. **P90 offers the best balance** between peak emphasis and selectivity — sufficient peak timesteps for training (~17%), improved zero-peak sequence fraction vs P85, and strong validation presence.
4. **P95 is viable but conservative** — better discrimination properties, yet fewer peak timesteps may limit the impact of peak-aware weighting; useful as a robustness alternative.
5. **Sequence-level weighting is ineffective for all three thresholds** (Task 3) — Phase 3 should use **timestep-level weighting** regardless of which percentile is chosen.
6. **Near-idle containers** affect all thresholds but most strongly at P95 (37% of peak labels); this is a Task 7 discussion point, not a reason to change exploration methodology.

#### Preliminary Recommendation (for Task 7)

**P90 per-container percentile on real CPU %** remains the **leading candidate**, based on:

- Balanced peak frequency on train and validation
- Standard interpretability (top 10% per container)
- Acceptable zero-peak sequence discrimination
- Consistent feasibility across all Phase 2 checks

P95 is the strongest alternative if Task 7 prioritizes stricter peak semantics and better near-idle robustness.

#### Limitations

- Comparison synthesizes exploratory statistics; no forecasting experiments have been run yet.
- Workload stratification for spiky containers is distorted by near-idle mislabelling.
- Final selection should be formalized in Task 7 with explicit rejection rationale.

#### Impact on Task 7

Task 7 should formally select one threshold (likely P90), document rejected alternatives with evidence from this table, discuss near-idle limitations, and confirm timestep-level weighting for Phase 3.

#### Next Task

**Task 7 — Peak Definition Decision** (awaiting approval).

---

### Task 7 — Peak Definition Decision

**Status:** Complete  
**Date:** 2026-07-14

#### Objective

Formally select and lock **one** peak definition for the remainder of the Peak-Aware Hybrid research, based on evidence from Tasks 2–6. Document rejected alternatives, limitations, and implications for Phase 3 (Peak-Aware Learning Design).

#### Decision

##### Chosen peak definition (LOCKED)

| Parameter | Specification |
|-----------|---------------|
| **Percentile** | **P90** (90th percentile) |
| **Scope** | Per-container |
| **Fit period** | Train period only |
| **Value space** | Real CPU utilization percent |
| **Derivation** | Inverse MinMax transform of `cpu_scaled` per container |
| **Peak timestep rule** | `cpu_real >= P90(container_train)` |
| **Absolute CPU floor** | **Not applied** |

**Formal rule:**

> A timestep is a peak if the container's real CPU utilization (%) is greater than or equal to that container's 90th-percentile CPU value computed on the train period only.

This definition is **locked** for all subsequent Peak-Aware training, evaluation labelling, and thesis reporting.

#### Justification

| Criterion | Evidence (Tasks 2–6) |
|-----------|---------------------|
| Sufficient peak frequency | ~17% train / ~28% val peak timesteps — adequate for peak-aware learning without diluting the label |
| Not too permissive | P85 at ~22% train peaks labels too many timesteps; weaker "spike" semantics |
| Not too strict | P95 at ~11% train peaks may under-emphasize moderate surges |
| Validation feasibility | 98.99% of containers have ≥1 peak in Day-1 horizon; 25.65% Day-1 peak timesteps |
| Standard interpretability | 90th percentile is widely understood as "unusually high for this container" |
| Sequence discrimination | Slightly better zero-peak sequence rate than P85 (3.8% vs 3.4%) |
| Consistency across tasks | Leading candidate across timestep, sequence, workload, and validation analyses |

#### Rejected Alternatives

| Alternative | Reason for rejection |
|-------------|---------------------|
| **P85** | Too loose (~21.7% train peaks, ~33.9% val); only 3.4% zero-peak sequences; difficult to defend as "spike" in thesis |
| **P95** | Too strict (~11.1% train peaks); highest near-idle distortion (37% of peak labels); viable but conservative — retained as robustness note only |
| **Global threshold** | Invalid under per-container MinMax scaling across heterogeneous workloads |
| **Absolute CPU floor** (e.g. ≥20%) | Considered; affects only 5/99 near-idle containers; would add complexity without Phase 2 evidence that P90 fails for the main cohort; **documented as limitation, not applied** |
| **Residual-based peaks** | Not explored as primary; CPU-real peaks are more operationally interpretable for cloud deployment |
| **Sequence-level weighting** | Rejected for Phase 3 implementation; >92% of sequences are peak sequences under all percentiles (Task 3) |

#### Expected Characteristics Under Locked Definition

| Metric | Train (expected) | Validation (expected) |
|--------|------------------|----------------------|
| Peak timestep % | ~17.4% | ~28.3% |
| Peak sequence % | ~96.2% | N/A (use Day-1 horizon) |
| Zero-peak sequence % | ~3.8% | — |
| Mean peak steps / sequence | ~16.3 of 96 | — |
| Day-1 containers with ≥1 peak | — | ~99.0% |
| Day-1 peak timestep % | — | ~25.7% |

#### Expected Impact on Phase 3 (Peak-Aware Learning Design)

| Phase 3 decision | Recommendation (from Phase 2 evidence) |
|------------------|----------------------------------------|
| **Weighting granularity** | **Timestep-weighted MSE** — sequence weighting is ineffective when >96% of sequences contain peaks |
| **Early stopping** | **Unweighted `val_loss`** — same as baseline; only training loss is weighted |
| **Peak weight λ** | Simple justification in Phase 3; suggested starting point **λ = 5** (moderate emphasis without dominating MSE) |
| **Implementation** | Parallel modules only; baseline frozen |
| **Evaluation** | Overall + peak-only + non-peak metrics using this same P90 definition on validation |

#### Limitations (to report in thesis)

1. **Near-idle zero-threshold artefact:** Five containers (`c_13308`, `c_14106`, `c_15035`, `c_15446`, `c_15794`) have P90 threshold = 0.0, contributing ~29% of train peak labels at P90 despite representing 5% of timesteps.
2. **Train/validation rate divergence:** Validation peak rates (~28%) exceed train rates (~17%) under fixed train thresholds — possible distribution shift between chronological splits.
3. **Spiky workload mislabelling:** High-CV near-idle containers inflate spiky-group statistics (Task 4).
4. **Short train history:** ~614 steps per container limits percentile precision for low-activity workloads.
5. **Preprocessing smoothing:** 15-minute resampling and interpolation may attenuate very brief spikes.

#### Generated Outputs

| Output | Location |
|--------|----------|
| Machine-readable decision | `peak_definition_decision.json` |
| Decision memo | `peak_definition_decision.md` |

#### Conclusion

Phase 2 is **complete**. The locked peak definition is **per-container P90 on real CPU %, train-fitted thresholds only**. Phase 3 (Peak-Aware Learning Design) may proceed.

---

## Phase 2 Summary

Phase 2 Peak Exploration ran from 2026-07-14 across seven tasks:

| Task | Outcome |
|------|---------|
| 1. Workspace | Experiment directory and research log established |
| 2. Timestep analysis | P85/P90/P95 compared; near-idle artefact identified |
| 2 supplement | Near-idle impact quantified (5% timesteps, 29% peak labels at P90) |
| 3. Sequence analysis | >92% peak sequences; timestep weighting recommended |
| 4. Workload stratification | Spiky elevation largely near-idle-driven |
| 5. Validation sanity | All percentiles feasible; val peaks ≥ train peaks |
| 6. Threshold comparison | P90 selected as leading candidate |
| 7. Peak definition | **P90 locked** |

**Next phase:** Phase 3 — Peak-Aware Learning Design (awaiting approval to begin).

