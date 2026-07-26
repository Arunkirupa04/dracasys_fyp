# Complete Preprocessing Guide: From Raw Data to Training Ready

## Overview

Module 2's preprocessing pipeline transforms raw AIOpsArena container metrics into training-ready data through 3 sequential steps. This guide explains **what each step does**, **why it's necessary**, and **how to execute it**.

---

## Architecture Overview

```
Step 0: Verify Raw Data
    ↓
Step 1: Merge Metrics into Case Files
    (WHY: Consolidate 32 separate metric files into 4 case files)
    ↓
Step 2: Normalize & Merge All Cases
    (WHY: Apply per-case normalization and create continuous dataset)
    ↓
Step 3: Add Per-Container Normalization
    (WHY: Handle 8.4x container volatility difference)
    ↓
OUTPUT: training_data_with_container_norm.csv
    ↓
Ready for Phase 1: Sequence Generation
```

---

## Step 0: Verify Raw Data Structure

### What You Need

Before starting, ensure raw data exists at:

```
module2/data/raw/
├── single/
│   ├── case1/container/*.csv (8 metrics)
│   └── case2/container/*.csv (8 metrics)
└── complex/
    ├── case1/container/*.csv (8 metrics)
    └── case2/container/*.csv (8 metrics)
```

**Total files needed:** 4 scenarios × 8 metrics = 32 CSV files

### 8 Required Metrics

Each case must have these metric files:

1. `kpi_container_cpu_usage_seconds_total.csv` - CPU total usage
2. `kpi_container_cpu_system_seconds_total.csv` - Kernel CPU time
3. `kpi_container_cpu_user_seconds_total.csv` - User CPU time
4. `kpi_container_memory_usage_bytes.csv` - Memory consumption
5. `kpi_container_memory_working_set_bytes.csv` - Active memory pages
6. `kpi_container_memory_rss.csv` - Resident set size
7. `kpi_container_memory_cache.csv` - Cached memory pages
8. (Optional) Additional metrics as needed

### Verify Data Exists

```bash
# Check file count
Get-ChildItem -Recurse -Filter "*.csv" | Measure-Object

# Should return: Count: 32 (or more if extra metrics exist)
```

---

## Step 1: Merge Raw Metrics into Case-Specific CSV Files

### Purpose

**Why this step?**

The raw AIOpsArena data stores each metric in a separate CSV file. For training, we need:
- All 7 metrics combined for each container and timestamp
- Single row per (timestamp, container_id) pair
- Standardized container ID mapping (cmdb_id → container_1 through container_27)

This is the **data consolidation step** that prepares for analysis.

### What Happens

**Input:** 32 separate metric CSV files
```
data/raw/
├── single/case1/container/kpi_container_cpu_usage_seconds_total.csv
├── single/case1/container/kpi_container_cpu_system_seconds_total.csv
├── ... (6 more metrics)
├── complex/case2/container/kpi_container_memory_cache.csv
└── ... (and so on for all 32 files)
```

**Process:**

For each of the 4 cases (single/case1, single/case2, complex/case1, complex/case2):

1. **Load all 7 metric files** for that case
   - Each file has columns: `timestamp`, `cmdb_id`, `value`
   - cmdb_id looks like: `observe.cartservice-0`, `observe.checkoutservice-1`, etc.

2. **Merge on (timestamp, cmdb_id)**
   - Create one row per container per timestamp
   - Use outer join to preserve all records
   - Handle missing values (if any metric missing at a timestamp)

3. **Assign container IDs**
   - Create mapping: unique cmdb_id → container_1, container_2, ..., container_27
   - Same mapping used across all 4 cases
   - This standardization is important for cross-case analysis

4. **Validate data quality**
   - Check timestamp continuity (15-second intervals)
   - Verify all 27 containers present
   - Confirm no unexpected null values

5. **Save merged CSV**
   - Columns: `timestamp`, `cmdb_id`, `new_container_id`, + 7 metrics
   - File: `module2/data/processed/{case}_merged.csv`
   - Size: ~74k to ~318k rows depending on case

**Output:** 4 CSV files
```
data/processed/
├── single_case1_merged.csv      (74,223 rows × 10 columns)
├── single_case2_merged.csv      (45,738 rows × 10 columns)
├── complex_case1_merged.csv     (318,445 rows × 10 columns)
└── complex_case2_merged.csv     (94,932 rows × 10 columns)
```

### How to Execute

```bash
cd C:\Users\DELL\Documents\Claude\Projects\short-term-prediction\module2
python preprocessing/run_merge_cases.py
```

**Expected output:**
```
======================================================================
MODULE 2: MERGE ALL CASES
======================================================================
Using config: ...data_config.yaml
Loaded config from ...data_config.yaml

Processing Case: single_case1
  ✓ Merged single/case1: 74,223 rows, 10 columns
  ✓ Assigned 27 unique container IDs
  ✓ Saved single_case1_merged.csv

Processing Case: single_case2
  ✓ Merged single/case2: 45,738 rows, 10 columns
  ✓ Assigned 27 unique container IDs
  ✓ Saved single_case2_merged.csv

Processing Case: complex_case1
  ✓ Merged complex/case1: 318,445 rows, 10 columns
  ✓ Assigned 27 unique container IDs
  ✓ Saved complex_case1_merged.csv

Processing Case: complex_case2
  ✓ Merged complex/case2: 94,932 rows, 10 columns
  ✓ Assigned 27 unique container IDs
  ✓ Saved complex_case2_merged.csv

SUCCESS: All cases merged!
```

**Time:** ~30 seconds

### Why This Matters

- **Consolidation:** Reduces 32 files → 4 files
- **Standardization:** Consistent column order and naming
- **Container mapping:** Enables cross-case container tracking
- **Data quality:** Early validation catches issues

---

## Step 2: Normalize & Merge All Cases with Continuous Timestamps

### Purpose

**Why this step?**

Now we have 4 separate case files, but for GRU training we need:
- **One continuous dataset** with all cases concatenated
- **Normalized data** using z-score (each metric has mean=0, std=1)
- **Consistent timestamps** without gaps between cases
- **Training-ready format** with proper column structure

This is the **normalization & unification step**.

### What Happens

**Input:** 4 merged CSV files from Step 1
```
data/processed/
├── single_case1_merged.csv
├── single_case2_merged.csv
├── complex_case1_merged.csv
└── complex_case2_merged.csv
```

**Process:**

1. **Load each case file**
   - Read in order: complex_case2, single_case2, single_case1, complex_case1
   - This order is chronologically based on original timestamps

2. **Apply per-case z-score normalization**
   
   For each case independently:
   ```
   For each metric M:
     case_mean = mean(M values in this case)
     case_std = std(M values in this case)
     normalized_M = (M - case_mean) / case_std
   ```
   
   **Why per-case normalization?**
   - Each case may have different baseline resource usage
   - Per-case normalization handles case-level drift
   - All 27 containers within a case use same statistics
   - Enables drift detection across cases (detection happens later)

3. **Reassign timestamps for continuity**
   
   Remove gaps between cases:
   - complex_case2: uses original timestamps (1719223200 onwards)
   - single_case2: starts immediately after complex_case2 ends
   - single_case1: starts immediately after single_case2 ends
   - complex_case1: starts immediately after single_case1 ends
   
   **Why reassign?**
   - Original datasets collected at different times with gaps
   - GRU expects continuous time series
   - Artificial continuous timestamps enable sequence generation
   - Preserves temporal relationships within each case

4. **Concatenate all 4 cases**
   - Stack them vertically into single DataFrame
   - Preserve container IDs and other mappings
   - Final dataset: 533,338 rows

5. **Save outputs**
   - Main CSV: `training_data_normalized_merged.csv` (~220 MB)
   - Stats JSON: `normalization_stats.json` (for denormalization later)

**Output:** 2 files
```
data/merged/
├── training_data_normalized_merged.csv
│   Columns: timestamp | case_source | cmdb_id | new_container_id | +7 normalized metrics
│   Rows: 533,338
│   Size: ~220 MB
│
└── normalization_stats.json
    {
      "single_case1": {
        "cpu_usage": {"mean": 125.43, "std": 45.2},
        "cpu_system": {...},
        ...
      },
      "single_case2": {...},
      ...
    }
```

### How to Execute

```bash
cd C:\Users\DELL\Documents\Claude\Projects\short-term-prediction\module2
python preprocessing/run_normalize_and_merge.py
```

**Expected output:**
```
======================================================================
MODULE 2: NORMALIZE & MERGE WITH CONTINUOUS TIMESTAMPS
======================================================================

SUCCESS: Normalization and merging complete!
======================================================================

Output files:
  ✓ Merged data: training_data_normalized_merged.csv
  ✓ Stats file: normalization_stats.json

Data summary:
  Total rows: 533,338
  Cases merged: 4
```

**Time:** ~5 seconds

### Why This Matters

- **Normalization:** Enables GRU to learn patterns (not absolute values)
- **Continuity:** Removes gaps for proper sequence generation
- **Standardization:** Mean=0, Std=1 across all metrics
- **Reproducibility:** Stats saved for denormalization during inference

---

## Step 3: Add Per-Container Normalization

### Purpose

**Why this step?**

Analysis shows container volatility varies **8.4x** across our 27 containers:
- Low-volatility container: std = 0.15
- High-volatility container: std = 1.26
- Wide range: 8.4x difference

**Per-case normalization isn't enough** because it treats all containers equally. We need:
- **Per-container statistics** to handle individual container behavior
- **Additional normalized columns** alongside per-case normalized ones
- **Fine-grained baseline removal** for accurate drift detection

This is the **container-specific normalization step**.

### Container Volatility Analysis

```
Container Examples:
  - cartservice (stable): std = 0.18 (low volatility)
  - currencyservice (variable): std = 0.89 (medium volatility)
  - checkoutservice (spiky): std = 1.26 (high volatility)
```

**Problem:** Per-case normalization treats these equally, but they have very different behavior patterns.

**Solution:** Add per-container normalized columns so GRU can learn container-specific patterns.

### What Happens

**Input:** `training_data_normalized_merged.csv` from Step 2

**Process:**

1. **Calculate per-container statistics**
   
   For each of 27 containers, for each of 7 metrics:
   ```
   container_mean = mean(metric values for this container)
   container_std = std(metric values for this container)
   container_min = min(metric values for this container)
   container_max = max(metric values for this container)
   ```
   
   **Result:** 189 statistics sets (27 containers × 7 metrics)

2. **Create per-container normalized columns**
   
   For each metric, add new column:
   ```
   {metric}_norm_container = (value - container_mean) / container_std
   ```
   
   **Examples:**
   - `container_cpu_usage_seconds_total_norm_container`
   - `container_memory_usage_bytes_norm_container`
   - ... (7 total new columns)
   
   **Result:** Each container gets N(0,1) distribution for each metric

3. **Validate normalization**
   - Check each container's normalized mean ≈ 0 (should be <0.01)
   - Check each container's normalized std ≈ 1 (should be 0.99-1.01)
   - Verify no NaN values introduced

4. **Save outputs**
   - Extended CSV: `training_data_with_container_norm.csv` (~400 MB)
   - Stats JSON: `per_container_stats.json`

**Output:** 2 files
```
data/merged/
├── training_data_with_container_norm.csv
│   Columns: (Step 2 columns) + 7 per-container normalized columns
│   Columns: timestamp | case_source | cmdb_id | new_container_id | 
│             cpu_usage | cpu_usage_norm_container |
│             memory_usage | memory_usage_norm_container |
│             ... (total 24 columns)
│   Rows: 533,338
│   Size: ~400 MB
│
└── per_container_stats.json
    {
      "container_1": {
        "cpu_usage": {"mean": 125.43, "std": 45.2, "min": 0, "max": 500},
        ...
      },
      ...
      "container_27": {...}
    }
```

### How to Execute

```bash
cd C:\Users\DELL\Documents\Claude\Projects\short-term-prediction\module2
python preprocessing/run_per_container_norm.py
```

**Expected output:**
```
======================================================================
MODULE 2: ADD PER-CONTAINER NORMALIZATION
======================================================================

SUCCESS: Per-container normalization complete!
======================================================================

Output files:
  ✓ Normalized data: training_data_with_container_norm.csv
  ✓ Stats file: per_container_stats.json

Data summary:
  Total rows: 533,338
  Containers: 27
  Metrics normalized: 7

Ready for Phase 1: Sequence Generation ✓
```

**Time:** ~30 seconds

### Why This Matters

- **Container-aware:** Handles 8.4x volatility difference
- **Dual normalization:** Both per-case and per-container data available
- **GRU flexibility:** Can learn from either normalization as needed
- **Drift detection:** Per-container stats enable container-specific thresholds

---

## Output Data Structure After All Steps

### Final Dataset: training_data_with_container_norm.csv

```
timestamp | case_source | cmdb_id | new_container_id |
cpu_usage | cpu_usage_norm_container |
cpu_system | cpu_system_norm_container |
cpu_user | cpu_user_norm_container |
memory_usage | memory_usage_norm_container |
memory_working_set | memory_working_set_norm_container |
memory_rss | memory_rss_norm_container |
memory_cache | memory_cache_norm_container
```

**Total:** 24 columns, 533,338 rows

### Column Breakdown

**Identifiers (4 columns):**
- `timestamp` - Unix epoch (continuous, no gaps)
- `case_source` - Which case (single/case1, etc.)
- `cmdb_id` - Original container ID
- `new_container_id` - Standardized ID (container_1 to container_27)

**Per-case normalized metrics (7 columns):**
- `container_cpu_usage_seconds_total` - Normalized using per-case stats
- `container_cpu_system_seconds_total`
- `container_cpu_user_seconds_total`
- `container_memory_usage_bytes`
- `container_memory_working_set_bytes`
- `container_memory_rss`
- `container_memory_cache`

**Per-container normalized metrics (7 columns):**
- `{metric}_norm_container` - Normalized using per-container stats
- One for each metric above

### Data Statistics

```
Total rows: 533,338 (samples)
Unique containers: 27
Total duration: ~61 hours (continuous timestamps)
Sampling interval: 15 seconds
Total size: ~400 MB

Breakdown by case:
  - complex_case2: 94,932 rows
  - single_case2: 45,738 rows
  - single_case1: 74,223 rows
  - complex_case1: 318,445 rows
```

---

## Execution Checklist

### Before Starting

```
☐ Verify raw data exists:
  module2/data/raw/single/case{1,2}/container/*.csv
  module2/data/raw/complex/case{1,2}/container/*.csv
  
☐ Verify config file exists:
  module2/config/data_config.yaml
  
☐ Verify runner scripts exist:
  module2/preprocessing/run_merge_cases.py
  module2/preprocessing/run_normalize_and_merge.py
  module2/preprocessing/run_per_container_norm.py
```

### Execution Steps

```
☐ Step 1: Run merge_cases
  cd C:\Users\DELL\Documents\Claude\Projects\short-term-prediction\module2
  python preprocessing/run_merge_cases.py
  
  Expected output:
  - module2/data/processed/ folder created
  - 4 CSV files: *_merged.csv
  - Time: ~30 seconds

☐ Step 2: Run normalize_and_merge
  python preprocessing/run_normalize_and_merge.py
  
  Expected output:
  - module2/data/merged/ folder created
  - training_data_normalized_merged.csv
  - normalization_stats.json
  - Time: ~5 seconds

☐ Step 3: Run per_container_norm
  python preprocessing/run_per_container_norm.py
  
  Expected output:
  - training_data_with_container_norm.csv
  - per_container_stats.json
  - Time: ~30 seconds

☐ Verify outputs
  dir module2\data\merged\
  
  Should show:
  - training_data_normalized_merged.csv
  - training_data_with_container_norm.csv
  - normalization_stats.json
  - per_container_stats.json
```

### Verification

```bash
# Check file sizes
ls -lh module2/data/merged/

# Should show:
# -rw-r--r-- 220 MB training_data_normalized_merged.csv
# -rw-r--r-- 400 MB training_data_with_container_norm.csv
# -rw-r--r--  40 KB normalization_stats.json
# -rw-r--r--  50 KB per_container_stats.json
```

---

## Common Issues & Solutions

### Issue: "File not found: data/raw/..."

**Cause:** Running from wrong directory

**Solution:**
```bash
# ✓ Correct - run from module2
cd C:\Users\DELL\Documents\Claude\Projects\short-term-prediction\module2
python preprocessing/run_merge_cases.py

# ✗ Wrong - don't run from preprocessing
cd preprocessing
python run_merge_cases.py  # Path resolution fails!
```

### Issue: "config/data_config.yaml not found"

**Cause:** Config file path is incorrect

**Solution:**
- Verify file exists at: `module2/config/data_config.yaml`
- Run from `module2` directory

### Issue: "Permission denied" on /merged folder

**Cause:** Folder already exists with restricted permissions

**Solution:**
```bash
# Delete and let script recreate
rmdir module2\data\merged /s /q
python preprocessing/run_normalize_and_merge.py
```

### Issue: Script runs but output files not created

**Cause:** Process may have failed silently

**Solution:**
- Check output folder for partial files
- Verify disk space available (need ~500 MB free)
- Run with verbose output:
  ```bash
  python preprocessing/run_merge_cases.py 2>&1 | tee output.log
  ```

---

## Next Steps: Phase 1 - Sequence Generation

Once preprocessing completes, use `training_data_with_container_norm.csv` for:

1. **Sliding Window Creation**
   - Create sequences of 240 timesteps (= 1 hour at 15-sec intervals)
   - Per container, using per-container normalized columns
   - Stride: 15 timesteps (overlapping windows)

2. **Target Generation**
   - Create 1-12 step ahead targets
   - For each timestep, predict next 1, 2, 3, ... 12 steps ahead
   - Each target is (batch_size, future_steps, metrics)

3. **Train/Val/Test Split**
   - Complex_case2 → Validation
   - Single_case2 → Test
   - Single_case1 + Complex_case1 → Training

4. **Model Input**
   - GRU model receives:
     - Input sequences: (batch_size, 240 timesteps, 7 per-container-normalized metrics)
     - Output targets: (batch_size, 1-12 future steps, 7 metrics)

---

## Summary

| Step | Input | Output | Time | Key Purpose |
|------|-------|--------|------|-------------|
| **Step 1: Merge** | 32 metric CSVs | 4 case CSVs | ~30s | Consolidate metrics, assign container IDs |
| **Step 2: Normalize & Merge** | 4 case CSVs | 1 merged CSV + stats | ~5s | Per-case normalization, continuous timestamps |
| **Step 3: Container Norm** | Merged CSV | Extended CSV + stats | ~30s | Per-container normalization for volatility handling |
| **Total** | Raw data | **Training-ready data** | **~1 min** | **Ready for Phase 1** |

---

## Key Concepts

### Normalization Strategy

```
Per-case normalization:
  - Handles case-level differences
  - All 27 containers get same statistics within a case
  - Used in Step 2
  
Per-container normalization:
  - Handles 8.4x container volatility
  - Each container gets unique statistics
  - Added in Step 3
  
Result: GRU has access to both, can learn from optimal representation
```

### Container Volatility Problem

```
Without per-container normalization:
  - High-volatility containers dominate learning
  - GRU learns worst-case patterns
  - Low-volatility containers' small changes missed

With per-container normalization:
  - Each container centered at 0, std at 1
  - All containers equally important
  - GRU learns container-specific patterns
  - Drift detection per-container (later phase)
```

### Timestamp Continuity

```
Original data (with gaps between cases):
  complex_case2: t0 to t1 (timestamps from original data)
  [GAP]
  single_case2: t0 to t2 (different time period)
  [GAP]
  ...

After preprocessing (continuous):
  complex_case2: t0 to t1 (original timestamps)
  single_case2: t1+1 to t3 (reassigned, no gap)
  single_case1: t3+1 to t4 (reassigned, no gap)
  complex_case1: t4+1 to t5 (reassigned, no gap)
  
Benefit: GRU expects continuous time series, no gaps in data
```

---

**Preprocessing Complete!** You now have `training_data_with_container_norm.csv` ready for Phase 1: Sequence Generation.
