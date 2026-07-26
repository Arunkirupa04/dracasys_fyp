# From Scratch Workflow: Complete Pipeline

## Raw Data to Ready-for-Training in 4 Steps

If you start with ONLY the raw AIOpsArena data, here's exactly what to run:

---

## Step 0: Verify Raw Data Structure

Before running anything, ensure you have:

```
module2/data/raw/
├── single/
│   └── case2/
│       └── container/
│           ├── kpi_container_cache.csv
│           ├── kpi_container_cpu_system_seconds_total.csv
│           ├── kpi_container_cpu_user_seconds_total.csv
│           ├── kpi_container_cpu_usage_seconds_total.csv
│           ├── kpi_container_memory_cache.csv
│           ├── kpi_container_memory_rss.csv
│           ├── kpi_container_memory_usage_bytes.csv
│           └── kpi_container_memory_working_set_bytes.csv
│
└── complex/
    └── case2/
        └── container/
            ├── kpi_container_cache.csv
            ├── ... (same 8 metrics)
            
(Plus single_case1, complex_case1 in same structure)
```

**Total:** 4 cases × 1 scenario (complex/single) × 8 metric files = 32 CSV files

---

## Step 1: Merge Raw Metrics into Case CSVs

**File to run:**
```
module2/preprocessing/merge_cases.py
```

**Via runner script (IMPORTANT: Run from module2, not preprocessing):**
```bash
cd C:\Users\DELL\Documents\Claude\Projects\short-term-prediction\module2
python preprocessing/run_merge_cases.py
```

**What it does:**
```
INPUT:  32 separate metric CSV files
  ├─ complex_case1/container/kpi_container_*.csv (8 files)
  ├─ complex_case2/container/kpi_container_*.csv (8 files)
  ├─ single_case1/container/kpi_container_*.csv (8 files)
  └─ single_case2/container/kpi_container_*.csv (8 files)

PROCESS:
  For each case:
    1. Load all 8 metric CSVs
    2. Merge on (timestamp, cmdb_id)
    3. Assign new_container_id (container_1 to container_27)
    4. Validate data quality
    5. Save merged CSV

OUTPUT: 4 CSV files
  module2/data/processed/
  ├─ complex_case1_merged.csv
  ├─ complex_case2_merged.csv
  ├─ single_case1_merged.csv
  └─ single_case2_merged.csv

SIZE: ~74k to ~318k rows each
```

**Log output:**
```
2024-06-24 15:30:00 - INFO - Processing complex_case2
2024-06-24 15:30:05 - INFO - Merged 8 metrics: 99,235 rows
2024-06-24 15:30:10 - INFO - Assigned container IDs: 1-27
2024-06-24 15:30:15 - INFO - Saved: complex_case2_merged.csv
...
SUCCESS: All 4 cases merged!
```

**Time:** ~30 seconds

---

## Step 2: Normalize & Merge Cases

**File to run:**
```
module2/preprocessing/normalize_and_merge.py
```

**Via runner script (IMPORTANT: Run from module2, not preprocessing):**
```bash
cd C:\Users\DELL\Documents\Claude\Projects\short-term-prediction\module2
python preprocessing/run_normalize_and_merge.py
```

**What it does:**
```
INPUT: 4 merged CSVs from Step 1
  ├─ complex_case2_merged.csv
  ├─ single_case2_merged.csv
  ├─ single_case1_merged.csv
  └─ complex_case1_merged.csv

PROCESS:
  1. Load each case
  2. Apply per-case z-score normalization
     - For each metric: (value - case_mean) / case_std
     - All 27 containers in case use same stats
  3. Reassign continuous timestamps
     - Remove gaps between cases
     - Start: 2024-06-24 15:30:00 (1719223200)
     - Interval: 15 seconds
  4. Concatenate all 4 cases
  5. Save merged dataset

OUTPUT: 2 files
  module2/data/merged/
  ├─ training_data_normalized_merged.csv (533,338 rows × 11 columns)
  └─ normalization_stats.json (per-case stats for denormalization)

COLUMNS:
  timestamp, case_source, cmdb_id, new_container_id, +7 metrics
```

**Log output:**
```
NORMALIZED AND MERGED RESULT
Total rows: 533,338
Total containers: 27
Cases included: 4
Timestamp range: 1719223200 → 1719446205
```

**Time:** ~5 seconds

**Output file size:** ~220 MB

---

## Step 3: Add Per-Container Normalization

**File to run:**
```
module2/preprocessing/add_per_container_normalization.py
```

**Via runner script (IMPORTANT: Run from module2, not preprocessing):**
```bash
cd C:\Users\DELL\Documents\Claude\Projects\short-term-prediction\module2
python preprocessing/run_per_container_norm.py
```

**What it does:**
```
INPUT: training_data_normalized_merged.csv from Step 2

PROCESS:
  1. Calculate per-container statistics
     - For each container + metric: mean, std, min, max
     - 27 containers × 7 metrics = 189 stats sets
  2. Create normalized columns
     - For each metric: {metric}_norm_container
     - Normalize to: (value - container_mean) / container_std
     - Each container gets its own N(0,1) distribution
  3. Validate normalization
     - Check each container's normalized mean ≈ 0
     - Check each container's normalized std ≈ 1

OUTPUT: 2 files
  module2/data/merged/
  ├─ training_data_with_container_norm.csv (new dataset, ~400 MB)
  └─ per_container_stats.json (per-container normalization stats)

COLUMNS: (Step 2 columns) + 7 new normalized columns
  {metric}_norm_container for each metric
```

**Log output:**
```
✓ Loaded 533,338 rows
✓ Calculated stats for 7 metrics × 27 containers
✓ Created 7 per-container normalized columns
✓ All validations passed
✓ Saved: training_data_with_container_norm.csv
✓ Saved: per_container_stats.json
```

**Time:** ~30 seconds

---

## Step 4: Verify with Analysis Notebooks (Optional)

**To verify everything worked:**

**Simple analysis:**
```bash
cd module2
jupyter notebook analysis_merged_data.ipynb
```

**Comprehensive analysis:**
```bash
cd module2
jupyter notebook analysis_merged_data_comprehensive.ipynb
```

**CPU variance deep-dive:**
```bash
cd module2
jupyter notebook cpu_variance_diagnostic.ipynb
```

**What they check:**
- ✓ All metrics present (7)
- ✓ All containers present (27)
- ✓ No null values
- ✓ Normalization applied correctly
- ✓ Continuous timestamps
- ✓ Per-container variation analyzed

---

## Complete From-Scratch Workflow (Visually)

```
RAW DATA
(32 CSV files in module2/data/raw)
     ↓
[1] cd module2
    python preprocessing/run_merge_cases.py
     ↓
PROCESSED DATA
(module2/data/processed: 4 merged CSVs)
     ↓
[2] python preprocessing/run_normalize_and_merge.py
     ↓
NORMALIZED + MERGED
(module2/data/merged: 1 CSV + 1 JSON)
     ↓
[3] python preprocessing/run_per_container_norm.py
     ↓
NORMALIZED + CONTAINER-NORMALIZED
(module2/data/merged: 1 new CSV + 1 new JSON)
     ↓
[4] (Optional) jupyter notebook analysis_merged_data_comprehensive.ipynb
     ↓
READY FOR SEQUENCE GENERATION (training_data_with_container_norm.csv)
```

**IMPORTANT:** Always run from `module2` directory, not from `preprocessing` folder!

---

## File Organization: Should You Reorganize?

### Current Structure
```
module2/
├── run_merge_cases.py              ← Runner for preprocessing/merge_cases.py
├── run_normalize_and_merge.py      ← Runner for preprocessing/normalize_and_merge.py
├── preprocessing/
│   ├── merge_cases.py              ← Implementation
│   ├── normalize_and_merge.py       ← Implementation
│   └── add_per_container_normalization.py
├── analysis_merged_data.ipynb
├── analysis_merged_data_comprehensive.ipynb
└── data/
    ├── raw/
    ├── processed/
    └── merged/
```

### Proposed Better Structure
```
module2/
├── preprocessing/
│   ├── run_merge_cases.py          ← MOVE HERE
│   ├── run_normalize_and_merge.py  ← MOVE HERE
│   ├── run_per_container_norm.py   ← ADD THIS (wrapper for add_per_container_normalization.py)
│   ├── merge_cases.py              ← Already here
│   ├── normalize_and_merge.py       ← Already here
│   └── add_per_container_normalization.py ← Already here
├── analysis/                        ← NEW FOLDER
│   ├── analysis_merged_data.ipynb   ← MOVE HERE
│   ├── analysis_merged_data_comprehensive.ipynb ← MOVE HERE
│   └── cpu_variance_diagnostic.ipynb ← MOVE HERE
├── notebooks/                       ← Alternative name for analysis/
└── data/
    ├── raw/
    ├── processed/
    └── merged/
```

### Advantages of Reorganization

**✓ Better organization:**
- All preprocessing code in one folder
- All analysis notebooks in one folder
- Clear separation of concerns

**✓ Easier to find files:**
- Runner scripts with implementations
- Notebooks grouped together

**✓ Simpler commands:**
```bash
cd module2/preprocessing
python run_merge_cases.py
python run_normalize_and_merge.py
python add_per_container_normalization.py
```

Instead of:
```bash
cd module2
python run_merge_cases.py
cd ..
cd preprocessing
python add_per_container_normalization.py
```

**✓ Cleaner root:**
- Only high-level files in module2/
- Execution is clearer

---

## Recommended Reorganization Steps

### 1. Create wrapper runner for Step 3
Create `module2/preprocessing/run_per_container_norm.py`:
```python
#!/usr/bin/env python3
"""Run per-container normalization"""

from add_per_container_normalization import PerContainerNormalizer
from pathlib import Path

script_dir = Path(__file__).parent.parent
merged_data_path = script_dir / "data" / "merged" / "training_data_normalized_merged.csv"
output_path = script_dir / "data" / "merged"

normalizer = PerContainerNormalizer(str(merged_data_path), str(output_path))
results = normalizer.process()
```

### 2. Move analysis notebooks
```bash
mkdir module2/analysis
mv module2/analysis_*.ipynb module2/analysis/
mv module2/cpu_variance_diagnostic.ipynb module2/analysis/
```

### 3. Move runner scripts
```bash
mv module2/run_merge_cases.py module2/preprocessing/
mv module2/run_normalize_and_merge.py module2/preprocessing/
```

### 4. Update imports if needed
- Check if runner scripts have hardcoded paths
- Update path references if moved

---

## New From-Scratch Command Sequence

After reorganization:

```bash
# All preprocessing in one place
cd module2/preprocessing

# Step 1: Merge cases
python run_merge_cases.py

# Step 2: Normalize & merge
python run_normalize_and_merge.py

# Step 3: Add per-container normalization
python run_per_container_norm.py

# Step 4: Analyze (from analysis folder)
cd ../analysis
jupyter notebook analysis_merged_data_comprehensive.ipynb
```

Much cleaner!

---

## Configuration File (data_config.yaml)

All paths defined in:
```
module2/config/data_config.yaml
```

Current structure:
```yaml
paths:
  raw_data_base_path: ./data/raw
  processed_data_path: ./data/processed
  merged_data_path: ./data/merged
  
cases:
  - complex_case1
  - complex_case2
  - single_case1
  - single_case2
  
metrics:
  - container_cpu_usage_seconds_total
  - (... 7 total)
```

If you reorganize folders, **only update paths in this YAML file**, not in code!

---

## Summary: From-Scratch Execution

### Minimum Commands (3 steps):
```bash
cd module2

# Step 1: Merge metrics by case
python run_merge_cases.py

# Step 2: Normalize & merge all cases
python run_normalize_and_merge.py

# Step 3: Add per-container normalization
python preprocessing/add_per_container_normalization.py
```

**Time:** ~1 minute total
**Output:** Ready for sequence generation

### With Reorganization:
```bash
cd module2/preprocessing

python run_merge_cases.py
python run_normalize_and_merge.py
python run_per_container_norm.py

# Then analyze
cd ../analysis
jupyter notebook analysis_merged_data_comprehensive.ipynb
```

---

## Answer to Your Questions

### Q1: What files do I need to run from scratch?

**Answer:** 3 main files:
1. `preprocessing/merge_cases.py` (via `run_merge_cases.py`)
2. `preprocessing/normalize_and_merge.py` (via `run_normalize_and_merge.py`)
3. `preprocessing/add_per_container_normalization.py`

Plus optional:
- Analysis notebooks (to verify)

### Q2: Can I move run_merge_cases.py and run_normalize_and_merge.py to preprocessing folder?

**Answer:** **YES, absolutely!** It's actually BETTER to do so.

**Benefits:**
- ✓ All preprocessing code in one place
- ✓ Clearer folder structure
- ✓ Easier to maintain
- ✓ Professional organization

**Steps:**
1. Move both runners to `module2/preprocessing/`
2. Create `module2/preprocessing/run_per_container_norm.py` (wrapper)
3. Move analysis notebooks to `module2/analysis/`
4. Update any hardcoded paths if needed
5. All 3 preprocessing steps run from `module2/preprocessing/`

**Recommendation:** Do this reorganization now before adding more code!

---

## Folder Structure After Reorganization

```
module2/
├── config/
│   └── data_config.yaml
│
├── preprocessing/               ← All preprocessing here
│   ├── run_merge_cases.py      ← Move from root
│   ├── run_normalize_and_merge.py ← Move from root
│   ├── run_per_container_norm.py  ← Create new
│   ├── merge_cases.py
│   ├── normalize_and_merge.py
│   └── add_per_container_normalization.py
│
├── analysis/                   ← All notebooks here
│   ├── analysis_merged_data.ipynb ← Move from root
│   ├── analysis_merged_data_comprehensive.ipynb ← Move from root
│   └── cpu_variance_diagnostic.ipynb ← Move from root
│
├── README.md                    ← Document structure
├── PROPOSAL_ROADMAP.md
├── ACTION_PLAN.md
└── data/
    ├── raw/
    ├── processed/
    └── merged/
```

This is production-ready structure!
