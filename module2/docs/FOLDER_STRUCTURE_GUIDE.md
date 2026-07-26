# Folder Structure & Path Management Guide

## Do You Need to Manually Create Folders?

### Short Answer: **NO**

The scripts **automatically create folders** if they don't exist.

```python
# From the scripts:
self.output_path.mkdir(parents=True, exist_ok=True)

# This means:
# ✓ Creates parent folders if missing
# ✓ Creates the folder if missing
# ✓ Does nothing if already exists (no error)
```

---

## Current Folder Structure

### What Should Already Exist

```
module2/
├── config/
│   └── data_config.yaml                    ← MUST EXIST
├── preprocessing/
│   ├── merge_cases.py                      ← MUST EXIST
│   ├── normalize_and_merge.py              ← MUST EXIST
│   └── add_per_container_normalization.py  ← MUST EXIST
└── data/
    └── raw/                                 ← MUST EXIST (with raw data)
        ├── single/
        │   └── case2/
        │       └── container/
        │           └── *.csv (8 metric files)
        ├── complex/
        │   └── case2/
        │       └── container/
        │           └── *.csv (8 metric files)
        └── ... (single_case1, complex_case1 in same structure)
```

**Critical:** `data/raw/` must exist with raw data files

### What Gets Created Automatically

```
module2/data/
├── raw/                    (manually created, has data)
├── processed/              ← CREATED BY merge_cases.py
│   ├── complex_case1_merged.csv
│   ├── complex_case2_merged.csv
│   ├── single_case1_merged.csv
│   └── single_case2_merged.csv
└── merged/                 ← CREATED BY normalize_and_merge.py
    ├── training_data_normalized_merged.csv
    ├── normalization_stats.json
    ├── training_data_with_container_norm.csv  (created by add_per_container_normalization.py)
    └── per_container_stats.json
```

---

## What Happens When You Run Scripts

### Step 1: merge_cases.py
```python
# Script checks if processed folder exists
if not processed_folder.exists():
    processed_folder.mkdir(parents=True, exist_ok=True)  # ✓ Creates it
# Then saves files there
```

**Result:** ✓ Creates `module2/data/processed/` if missing

### Step 2: normalize_and_merge.py
```python
# Script checks if merged folder exists
if not merged_folder.exists():
    merged_folder.mkdir(parents=True, exist_ok=True)  # ✓ Creates it
# Then saves files there
```

**Result:** ✓ Creates `module2/data/merged/` if missing

### Step 3: add_per_container_normalization.py
```python
# Same logic - creates merged folder if needed
# (but it already exists from Step 2)
```

**Result:** ✓ Saves new files in `module2/data/merged/`

---

## Summary: What You Need to Do

### Before Running Scripts

```
✓ Ensure this exists:
  module2/
  ├── config/data_config.yaml
  ├── preprocessing/
  │   ├── run_merge_cases.py              (Runner script)
  │   ├── run_normalize_and_merge.py      (Runner script)
  │   ├── run_per_container_norm.py       (Runner script)
  │   ├── merge_cases.py                  (Core module)
  │   ├── normalize_and_merge.py          (Core module)
  │   └── add_per_container_normalization.py  (Core module)
  └── data/raw/ (with raw CSV files)

✗ Do NOT manually create:
  ✗ data/processed/
  ✗ data/merged/
  (Scripts create these automatically)
```

### Running Scripts (IMPORTANT: From module2 directory)

**ALWAYS run from `module2` directory, NOT from preprocessing:**

```bash
cd C:\Users\DELL\Documents\Claude\Projects\short-term-prediction\module2

# Step 1: ✓ Creates data/processed/
python preprocessing/run_merge_cases.py

# Step 2: ✓ Creates data/merged/
python preprocessing/run_normalize_and_merge.py

# Step 3: ✓ Adds files to data/merged/
python preprocessing/run_per_container_norm.py
```

**Why from module2?** Relative paths in config (like `./data/raw`) resolve correctly when running from `module2/` directory. Running from `preprocessing/` causes path resolution to fail.

After this, all folders + files exist automatically!

---

## Moving Data Folder: Will It Break Paths?

### The Short Answer: **It Depends on data_config.yaml**

The scripts don't have hardcoded paths. They read from config file:

```
module2/config/data_config.yaml
```

---

## How Path Resolution Works

### Current Configuration

**File:** `module2/config/data_config.yaml`
```yaml
paths:
  raw_data_base_path: ./data/raw
  processed_data_path: ./data/processed
  merged_data_path: ./data/merged
```

**How scripts use it:**
```python
# In merge_cases.py
config = read_config('config/data_config.yaml')
raw_path = config['paths']['raw_data_base_path']
# Resolves to: module2/data/raw/
```

**Key:** Paths are **relative to module2/** (where scripts run from)

---

## Scenario 1: Keep data/ Folder Where It Is

```
Current:
  module2/
  └── data/
      ├── raw/
      ├── processed/  (created)
      └── merged/     (created)

Config: ./data/raw, ./data/processed, ./data/merged ✓ Works
```

**Result:** No changes needed!

---

## Scenario 2: Move data/ Folder Elsewhere

### Example: Move to C:/external_drive/container_data/

```
Before:
  module2/
  └── data/ ← Raw data here

After:
  module2/
  (no data/ folder)
  
  C:/external_drive/container_data/ ← Raw data moved here
  ├── raw/
  ├── processed/
  └── merged/
```

**Will it break?** **YES** ❌

**Why?** Scripts look for `./data/raw/` relative to module2/

**How to fix?** Update config file:

```yaml
# module2/config/data_config.yaml

paths:
  raw_data_base_path: C:/external_drive/container_data/raw
  processed_data_path: C:/external_drive/container_data/processed
  merged_data_path: C:/external_drive/container_data/merged
```

Now scripts find data at new location ✓

---

## Scenario 3: Move data/ Into Different Subfolder

### Example: Module structure change

```
Before:
  short-term-prediction/
  └── module2/
      └── data/

After:
  short-term-prediction/
  ├── module2/
  ├── module3/
  └── shared_data/  ← Move data here
      └── container_data/
          ├── raw/
          ├── processed/
          └── merged/
```

**Will it break?** **YES** ❌

**How to fix?**

**Option A:** Use absolute paths in config
```yaml
paths:
  raw_data_base_path: C:\Users\DELL\Documents\Claude\Projects\short-term-prediction\shared_data\container_data\raw
  processed_data_path: C:\Users\DELL\Documents\Claude\Projects\short-term-prediction\shared_data\container_data\processed
  merged_data_path: C:\Users\DELL\Documents\Claude\Projects\short-term-prediction\shared_data\container_data\merged
```

**Option B:** Use relative paths (from module2/)
```yaml
paths:
  raw_data_base_path: ../../shared_data/container_data/raw
  processed_data_path: ../../shared_data/container_data/processed
  merged_data_path: ../../shared_data/container_data/merged
```

Both work! ✓

---

## Best Practice: Keep Default Structure

### Recommended Approach

```
module2/
├── config/
│   └── data_config.yaml
├── preprocessing/
├── analysis/
├── data/                  ← Keep here (local to module)
│   ├── raw/              (raw AIOpsArena data)
│   ├── processed/        (auto-created by merge_cases.py)
│   └── merged/           (auto-created by normalize_and_merge.py)
└── ... (notebooks, runners, etc.)
```

**Advantages:**
- ✓ Default config works (no changes needed)
- ✓ Self-contained module (easy to move/backup)
- ✓ No path issues
- ✓ Cleaner project structure

**Config stays as-is:**
```yaml
paths:
  raw_data_base_path: ./data/raw
  processed_data_path: ./data/processed
  merged_data_path: ./data/merged
```

---

## Path Resolution: How It Works

### When You Run: `python run_merge_cases.py`

```
Current working directory: C:\Users\DELL\Documents\Claude\Projects\short-term-prediction\module2

Config path: ./config/data_config.yaml
Resolved to: C:\Users\DELL\Documents\Claude\Projects\short-term-prediction\module2\config\data_config.yaml ✓

Data path: ./data/raw
Resolved to: C:\Users\DELL\Documents\Claude\Projects\short-term-prediction\module2\data\raw ✓

Output path: ./data/processed
Resolved to: C:\Users\DELL\Documents\Claude\Projects\short-term-prediction\module2\data\processed ✓
```

**Key:** Everything is relative to where you run the command!

---

## Checklist: Before Running Scripts

```
□ module2/config/data_config.yaml exists?
  └─ Check paths point to correct locations

□ module2/preprocessing/ has all 3 scripts?
  ├─ merge_cases.py
  ├─ normalize_and_merge.py
  └─ add_per_container_normalization.py

□ module2/data/raw/ has raw CSV files?
  ├─ single/case2/container/*.csv (8 files)
  ├─ complex/case2/container/*.csv (8 files)
  ├─ single/case1/container/*.csv (8 files)
  └─ complex/case1/container/*.csv (8 files)

□ Run from module2/ directory?
  └─ cd C:\Users\DELL\Documents\Claude\Projects\short-term-prediction\module2

DO NOT create manually:
  ✗ data/processed/
  ✗ data/merged/
  (Scripts create these)
```

---

## If Something Goes Wrong: Troubleshooting

### Error: "raw_data_base_path not found"

**Cause:** Config points to wrong location

**Fix:**
```yaml
# Check module2/config/data_config.yaml
# Verify paths exist and have data

# Example correct path:
raw_data_base_path: ./data/raw
# Should resolve to: module2/data/raw/
```

### Error: "Permission denied" on processed/merged folders

**Cause:** Folder exists but you don't have write permission

**Fix:**
```bash
# Give full permissions
# On Windows:
icacls "module2\data\processed" /grant:r "%username%":F

# Or manually delete and let script recreate
rmdir module2\data\processed /s /q
# Script will recreate it automatically
```

### Error: "Cannot read normalization_stats.json"

**Cause:** Step 2 didn't complete successfully

**Fix:**
```bash
# Delete partial outputs
rmdir module2\data\merged /s /q

# Rerun Step 2
python run_normalize_and_merge.py

# It will recreate merged/ and all files
```

---

## Summary Table

| Question | Answer | Action |
|----------|--------|--------|
| Create data/processed/ manually? | NO | Scripts do it automatically |
| Create data/merged/ manually? | NO | Scripts do it automatically |
| Can I move data/ folder? | YES | Update data_config.yaml paths |
| What if I move to different location? | Update config | Use absolute or relative paths |
| Best practice location? | Keep at module2/data/ | Use default config |
| Do I need to run from module2/? | YES | cd module2/ before running |

---

## Example: Complete Setup from Scratch

### Step 1: Create directory structure
```bash
cd C:\Users\DELL\Documents\Claude\Projects\short-term-prediction\module2

# Verify these exist:
# ✓ config/data_config.yaml
# ✓ preprocessing/*.py (3 files)
# ✓ data/raw/ (with CSV files)

# DO NOT create:
# ✗ data/processed/
# ✗ data/merged/
```

### Step 2: Run scripts (creates folders automatically)
```bash
python run_merge_cases.py
# → Creates data/processed/ ✓

python run_normalize_and_merge.py
# → Creates data/merged/ ✓

python preprocessing/add_per_container_normalization.py
# → Uses existing data/merged/ ✓
```

### Step 3: Verify results
```bash
# Check created folders
dir module2\data\

# You should see:
# data/raw/            (had raw files)
# data/processed/      (created, has 4 CSVs)
# data/merged/         (created, has 3 files)
```

**Done!** ✓ All folders + files created automatically

---

## Final Answer

**Q: Do I need to manually create merged and processed folders?**

**A: NO**
- Scripts automatically create both folders
- Using: `mkdir(parents=True, exist_ok=True)`
- No manual folder creation needed

**Q: If moving data folder, will it affect paths?**

**A: YES - If you don't update config**
- Paths are defined in `data_config.yaml`
- Scripts read paths from config file
- If you move data folder:
  - Update config with new paths
  - Then scripts will find data at new location
  
**Best practice:** Keep `data/` folder inside `module2/`
- Uses default paths (no config changes needed)
- Self-contained, portable
- Simple and clean
