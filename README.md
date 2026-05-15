# Module 2: Short-Term Resource Prediction
## GRU-Based Drift-Aware Forecasting for Containerized Environments

---

## 📁 Folder Structure

```
module2/
├── README.md                           (This file)
├── __init__.py                         (Module initialization)
│
├── config/
│   ├── __init__.py
│   └── data_config.yaml                (Data paths and metrics configuration)
│
├── preprocessing/
│   ├── run_merge_cases.py              (Runner script - EXECUTE THIS FIRST)
│   ├── run_normalize_and_merge.py      (Runner script - EXECUTE THIS SECOND)
│   ├── run_per_container_norm.py       (Runner script - EXECUTE THIS THIRD)
│   ├── merge_cases.py                  (Merge all 4 cases into processed CSVs)
│   ├── normalize_and_merge.py          (Normalize and merge all cases)
│   └── add_per_container_normalization.py  (Add per-container normalization)
│
├── data/
│   ├── raw/                            (Link to AIOpsArena dataset)
│   │   ├── single/
│   │   │   ├── case1/container/*.csv
│   │   │   └── case2/container/*.csv
│   │   └── complex/
│   │       ├── case1/container/*.csv
│   │       └── case2/container/*.csv
│   │
│   └── processed/                      (OUTPUT: Merged CSV files)
│       ├── single_case1_merged.csv     (10 columns)
│       ├── single_case2_merged.csv     (10 columns)
│       ├── complex_case1_merged.csv    (10 columns)
│       └── complex_case2_merged.csv    (10 columns)
│
├── model/                              (Model architecture - future)
├── training/                           (Training pipeline - future)
├── inference/                          (Inference pipeline - future)
├── evaluation/                         (Evaluation - future)
└── results/                            (Output results - future)
```

---

## 🚀 Quick Start

### Step 1: Verify Configuration

Ensure `module2/config/data_config.yaml` is configured correctly:

```yaml
paths:
  raw_data_base_path: ./data/raw         # Path to AIOpsArena data
  processed_data_path: ./data/processed  # Output directory
  merged_data_path: ./data/merged        # Merged output directory
core_metrics: [7 core metrics]
cases: [single/case1, single/case2, complex/case1, complex/case2]
```

### Step 2: Run All 3 Preprocessing Scripts

```bash
cd C:\Users\DELL\Documents\Claude\Projects\short-term-prediction\module2

# Step 1: Merge cases (creates data/processed/)
python preprocessing/run_merge_cases.py

# Step 2: Normalize & merge (creates data/merged/)
python preprocessing/run_normalize_and_merge.py

# Step 3: Per-container normalization
python preprocessing/run_per_container_norm.py
```

### Step 3: Verify Output

After all 3 scripts complete, check:

**Step 1 output** - `module2/data/processed/`:
- `single_case1_merged.csv`
- `single_case2_merged.csv`
- `complex_case1_merged.csv`
- `complex_case2_merged.csv`

**Step 2 & 3 outputs** - `module2/data/merged/`:
- `training_data_normalized_merged.csv` (from Step 2)
- `normalization_stats.json` (from Step 2)
- `training_data_with_container_norm.csv` (from Step 3) ← **USE THIS for Phase 1**
- `per_container_stats.json` (from Step 3)

Merged CSV files should have **10 columns**:

```
timestamp | cmdb_id | new_container_id | cpu_usage | cpu_system | cpu_user | memory_usage | memory_working_set | memory_rss | memory_cache
```

---

## 📊 Output CSV Format

### Columns (10 total):

1. **timestamp** - Unix epoch time (int)
2. **cmdb_id** - Original container ID (string, e.g., "observe.cartservice-0")
3. **new_container_id** - Assigned unique ID (string, e.g., "container_1")
4. **container_cpu_usage_seconds_total** - CPU usage (float)
5. **container_cpu_system_seconds_total** - System CPU (float)
6. **container_cpu_user_seconds_total** - User CPU (float)
7. **container_memory_usage_bytes** - Memory usage (float)
8. **container_memory_working_set_bytes** - Active memory (float)
9. **container_memory_rss** - RSS memory (float)
10. **container_memory_cache** - Cache memory (float)

### Example Output:

```
timestamp,cmdb_id,new_container_id,container_cpu_usage_seconds_total,container_cpu_system_seconds_total,...
1719223200,observe.currencyservice-1,container_1,178.085,25.3,...
1719223200,observe.cartservice-0,container_2,110.141,24.1,...
1719223215,observe.currencyservice-1,container_1,180.234,25.5,...
...
```

---

## 🔍 Data Mapping Details

### Container ID Mapping

Original `cmdb_id` → Assigned `new_container_id`:

```
observe.cartservice-0        → container_1
observe.cartservice-1        → container_2
observe.cartservice-2        → container_3
observe.checkoutservice-0    → container_4
observe.checkoutservice-1    → container_5
...
(27 unique containers total)
```

### Timestamp Mapping

- **Source**: Unix epoch (seconds) from AIOpsArena
- **Format**: Integer timestamp
- **Precision**: 15-second intervals
- **Mapping**: No transformation, direct mapping from source

### Data Merging Strategy

For each case (single/case1, single/case2, complex/case1, complex/case2):

1. Load all 7 core metric CSV files
2. Merge on (timestamp, cmdb_id) using outer join
3. Assign unique new_container_id for each unique cmdb_id
4. Reorder columns: timestamp, cmdb_id, new_container_id, metrics
5. Sort by timestamp and cmdb_id
6. Save as CSV

---

## 📈 Data Statistics

After merging, each file contains:

| Case | Rows | Unique Containers | Duration |
|------|------|-------------------|----------|
| single_case1 | 60,615 | 27 | 9.3 hours |
| single_case2 | 38,934 | 27 | 6.0 hours |
| complex_case1 | 224,330 | 27 | 35.0 hours |
| complex_case2 | 77,922 | 27 | 12.0 hours |
| **TOTAL** | **401,801** | **27** | **62 hours** |

---

## 🛠️ Code Structure

### `merge_cases.py` - Main Processing Script

Key classes and methods:

```python
class CaseMerger:
    def __init__(config_path):
        """Initialize with config"""
    
    def _load_metric_csv(scenario, case, metric):
        """Load single metric CSV file"""
    
    def _merge_metrics_for_case(scenario, case):
        """Merge all 7 metrics for a case"""
    
    def _assign_container_ids(df):
        """Create mapping: cmdb_id -> new_container_id"""
    
    def _validate_data(df, scenario, case):
        """Validate merged data integrity"""
    
    def _prepare_output(df):
        """Prepare final CSV with correct column order"""
    
    def _save_case_data(df, scenario, case):
        """Save merged case to CSV file"""
    
    def process_all_cases():
        """Process all 4 cases and save"""
```

---

## ✅ Validation Checks

The script performs these validations:

- ✓ Verify all CSV files exist
- ✓ Check for required columns (timestamp, cmdb_id, value)
- ✓ Validate timestamp continuity
- ✓ Check for missing values
- ✓ Verify container count (27)
- ✓ Ensure proper merging on timestamp and cmdb_id
- ✓ Confirm new_container_id assignment is unique
- ✓ Validate output CSV structure

---

## 📝 Logging Output

When you run `python run_merge_cases.py`, you'll see:

```
2026-05-08 10:15:30 - INFO - Initialized CaseMerger
2026-05-08 10:15:31 - INFO - Processing single_case1...
2026-05-08 10:15:35 - INFO - Merged single_case1: 60,615 rows, 10 columns
2026-05-08 10:15:36 - INFO - Assigned 27 unique container IDs
2026-05-08 10:15:36 - INFO - Validating single_case1...
2026-05-08 10:15:37 - INFO - Saved single_case1_merged.csv
...
2026-05-08 10:15:50 - INFO - SUCCESS: All cases merged!
```

---

## 🔗 Integration with Next Steps

These 4 CSV files are used for:

### Phase 1: Normalization
```
single_case1_merged.csv → step2_normalizer.py
  → normalized per container
  → Save normalization_stats.json
```

### Phase 2: Sequence Creation
```
normalized_data.csv → step3_sequencer.py
  → Create sliding windows (240 timesteps)
  → Generate (X, y) pairs
  → Output: train_sequences.npz, val_sequences.npz, test_sequences.npz
```

### Phase 3: Training
```
sequences.npz → train.py
  → Train GRU model
  → Save gru_model.h5
```

### Phase 4: Inference & Drift Detection
```
gru_model.h5 + normalization_stats.json → predictor.py
  → Real-time predictions
  → Drift monitoring
```

---

## 🐛 Troubleshooting

### Issue: "File not found" errors

**Solution**: 
- Ensure AIOpsArena data is in `module2/data/raw/`
- Check folder structure: `raw/single/case{1,2}/container/*.csv`
- Verify all 7 metric CSV files exist

### Issue: "Config file not found"

**Solution**:
- Run from `module2` directory
- Ensure `config/data_config.yaml` exists
- Check path: `module2/config/data_config.yaml`

### Issue: "Null values in output"

**Solution**:
- Check if all metric CSVs have same number of rows
- Verify timestamps are aligned across metrics
- Check `handle_missing_values` in config

---

## 📚 Configuration Reference

### data_config.yaml Options

```yaml
# Handle missing values
handle_missing_values: "forward_fill"  # Options: forward_fill, interpolate, drop

# Validation
validate_timestamps: true
validate_cmdb_ids: true

# Output format
output_format: "csv"  # Only CSV supported currently
```

---

## 🎯 Next Steps

After running this script:

1. ✅ Verify 4 CSV files in `module2/data/processed/`
2. ✅ Check each file has 10 columns
3. ✅ Examine sample rows to verify mapping
4. 🔜 Run `step2_normalizer.py` for normalization
5. 🔜 Run `step3_sequencer.py` to create sequences
6. 🔜 Run `train.py` to train GRU model

---

## 📧 Questions?

Refer to:
- `MODULE2_INTEGRATION_PLAN.md` - Architecture overview
- `MODULE2_INTEGRATION_QUICK_REFERENCE.md` - Visual guide
- `data_config.yaml` - Configuration options
- Script logs - Detailed execution details

---

**Created**: 2026-05-08  
**Version**: 1.0  
**Status**: Ready for use
