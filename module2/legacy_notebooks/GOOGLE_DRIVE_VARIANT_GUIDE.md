# Google Drive Data Loading Variant - Complete Guide

**Notebook:** `Complete_Pipeline_Google_Drive_Data.ipynb`

**Purpose:** Load data directly from Google Drive and process through complete ML pipeline

**Best For:** Users who have data stored in Google Drive and want to avoid uploading to multiple locations

---

## 📍 Google Drive Folder Structure Required

Your Google Drive needs to have this exact nested structure:

```
My Drive/
└── raw/                          ← Main folder (MUST be named "raw")
    ├── complex/                  ← Case group folder
    │   ├── case1/                ← Case subfolder
    │   │   ├── istio/            ← (Optional - notebook skips this)
    │   │   └── container/        ← Container data folder
    │   │       ├── file1.csv
    │   │       ├── file2.csv
    │   │       └── ... (12-15 CSV files)
    │   │
    │   └── case2/                ← Case subfolder
    │       ├── istio/            ← (Optional - notebook skips this)
    │       └── container/        ← Container data folder
    │           ├── file1.csv
    │           ├── file2.csv
    │           └── ... (12-15 CSV files)
    │
    └── single/                   ← Case group folder
        └── case2/                ← Case subfolder
            ├── istio/            ← (Optional - notebook skips this)
            └── container/        ← Container data folder
                ├── file1.csv
                ├── file2.csv
                └── ... (12-15 CSV files)
```

**Key Points:**
- ✅ Main folder MUST be named `raw` (exactly)
- ✅ Structure: raw → {complex/single} → case{1/2} → {istio/container}
- ✅ CSV files are in the `container/` subfolder (notebook searches recursively)
- ✅ Istio folder is optional - notebook only loads from container folders
- ✅ Notebook automatically finds all CSV files at any nested depth
- ✅ CSV file naming doesn't matter

---

## 🚀 Quick Start (5 minutes)

### 1. Prepare Your Google Drive

```
✓ Create folder: My Drive/raw
✓ Create nested folders:
  - My Drive/raw/complex/case1/container/
  - My Drive/raw/complex/case2/container/
  - My Drive/raw/single/case2/container/
✓ Upload CSV files to each container/ folder
✓ Make sure folder is shared with your Google account
```

**Folder structure you should have:**
```
raw/
├── complex/
│   ├── case1/container/*.csv     (12-15 files)
│   └── case2/container/*.csv     (12-15 files)
└── single/
    └── case2/container/*.csv     (12-15 files)
```

### 2. Open Notebook in Google Colab

```
https://colab.research.google.com
↓
Upload the notebook: Complete_Pipeline_Google_Drive_Data.ipynb
(or paste it from your local Files folder)
```

### 3. Run Cells in Order

```
Cell 1 & 2: Check Colab environment
Cell 3:     Mount Google Drive
Cell 5:     Setup imports
Cell 6:     Configure paths
Cell 8:     Load data from Drive
Cell 10:    Normalize and split
Cell 12:    Feature engineering
Cell 14:    Define sequence generator
Cell 15:    Generate sequences
Cell 17:    Verify sequences
Cell 19:    Download/upload options
```

### 4. Download Results (Optional)

Sequences are saved in Colab storage at `/content/processed_data/sequences/`

- **Option A (Colab-only):** Keep in Colab for training
- **Option B (Local):** Download using Cell 19 code
- **Option C (Google Drive):** Save back to Google Drive using Cell 19 code

---

## 📋 Cell-by-Cell Breakdown

### Cell 1-2: Detect Environment
**What it does:** Checks if running in Google Colab

**Output you should see:**
```
✓ Running in Google Colab
Python version: 3.x.x
```

**If error:** Not running in Colab - open in https://colab.research.google.com

---

### Cell 3: Mount Google Drive
**What it does:** Connects Colab to your Google Drive

**Expected flow:**
1. Click "Run Cell" (play button)
2. See: "Mounting Google Drive..."
3. Click the link provided
4. Select your Google account
5. Copy the authorization code
6. Paste code into Colab input box
7. Press Enter

**Output you should see:**
```
✓ Google Drive mounted!
✓ Found raw folder at: /content/drive/My Drive/raw
Folder contents:
  📁 single/
     Found 5 CSV files
  📁 complex/
     Found 3 CSV files
```

**If error:** Check that:
- ✓ raw folder exists at: `My Drive/raw`
- ✓ Subfolders contain CSV files
- ✓ Folder is shared with your Google account

---

### Cell 5: Import Libraries
**What it does:** Loads pandas, numpy, pathlib, and logging

**Output you should see:**
```
✓ All imports successful
```

**If error:** Colab pre-installs all these - retry cell

---

### Cell 6: Configure Paths
**What it does:** Sets up paths and recursively searches for CSV files at ANY nested depth

**Paths created:**
- Input:  `/content/drive/My Drive/raw` (searched recursively with `**/*.csv`)
- Output: `/content/processed_data/processed/`
- Output: `/content/processed_data/merged/`
- Output: `/content/processed_data/sequences/`

**Output you should see:**
```
📁 Path Configuration:
  Google Drive Input:   /content/drive/My Drive/raw
  Local Processed:      /content/processed_data/processed
  Local Merged:         /content/processed_data/merged
  Local Sequences:      /content/processed_data/sequences

📊 Checking Google Drive folder structure:
✓ Raw folder accessible
✓ Found 39 total CSV files in nested structure:

Folder structure with CSV counts:

  📁 complex/case1/container/
     └─ 15 CSV files
        • file1.csv
        • file2.csv
        • file3.csv
        • ... and 12 more

  📁 complex/case2/container/
     └─ 14 CSV files
        • ...

  📁 single/case2/container/
     └─ 10 CSV files
        • ...
```

**Key point:** The notebook uses `raw.glob('**/*.csv')` to recursively find ALL CSV files at ANY depth, so it correctly locates them in:
- `raw/complex/case1/container/`
- `raw/complex/case2/container/`
- `raw/single/case2/container/`

---

### Cell 8: Load Data from Google Drive
**What it does:** Reads all CSV files from subfolders and combines them

**Expected flow:**
1. Iterates through each subfolder (single, complex)
2. Loads each CSV file
3. Combines all into one DataFrame

**Output you should see:**
```
Loading data from Google Drive: /content/drive/My Drive/raw

Processing subfolder: single
  Found 5 CSV files
  Loading: container_metrics_001.csv
    Shape: (10000, 45)
  Loading: machine_metrics_001.csv
    Shape: (5000, 30)
  ...

Processing subfolder: complex
  Found 3 CSV files
  ...

✓ Combined data shape: (140670, 45)
  Rows: 140,670
  Columns: 45

✓ Data loaded successfully

Data preview (first 5 rows):
[DataFrame preview]
```

---

### Cell 10: Split & Normalize
**What it does:** Prevents data leakage by splitting BEFORE normalizing

**Process:**
1. Sort by timestamp (if exists)
2. Split: 60% train, 20% val, 20% test
3. Normalize each split independently

**Output you should see:**
```
Splitting data chronologically...
Train: 84,402 rows (60.0%)
Val:   28,134 rows (20.0%)
Test:  28,134 rows (20.0%)

Normalizing each split independently...
  train: 40 columns normalized
  val: 40 columns normalized
  test: 40 columns normalized

✓ Data normalized
```

---

### Cell 12: Feature Engineering
**What it does:** Creates temporal features (lag differences, rolling statistics)

**Features created per target metric:**
- DIFF_1, DIFF_2, DIFF_3 (lag differences)
- ROLLING_MEAN_3 (3-step rolling average)
- ROLLING_STD_3 (3-step rolling standard deviation)

**Output you should see:**
```
======================================================================
FEATURE ENGINEERING
======================================================================

Available target columns: 4
  - container_cpu_usage_seconds_total
  - container_memory_usage_bytes
  - container_memory_working_set_bytes
  - container_memory_rss

Engineering features for train...
  Total columns after features: 64

Engineering features for val...
  Total columns after features: 64

Engineering features for test...
  Total columns after features: 64

✓ Features engineered and saved
```

---

### Cell 14: Define Sequence Generator
**What it does:** Creates the UltraFastSequenceGenerator class (memory-efficient)

**Key specifications:**
- Lookback window: 240 timesteps
- Horizons: 1-10 steps ahead
- Output format: NumPy binary (.npy)
- Memory usage: ~32 MB (vs. 8+ GB for full-load approach)

**Output you should see:**
```
✓ SequenceGenerator class ready
```

---

### Cell 15: Generate Sequences
**What it does:** Processes each dataset (train/val/test) into sequences

**Process:**
1. Groups by container_id
2. Creates 240-step lookback windows
3. Generates 1-10 step ahead targets
4. Saves in batches to avoid memory overflow
5. Creates metadata JSON files

**Expected runtime:** 5-10 minutes (depending on data size)

**Output you should see:**
```
######################################################################
PHASE 1: SEQUENCE GENERATION
######################################################################

Processing TRAIN data...
Creating sequences for train...
Processing 27 containers...
  Container 5/27... (valid: 15,000)
  Container 10/27... (valid: 45,000)
  Container 15/27... (valid: 90,000)
  Container 20/27... (valid: 120,000)
  Container 25/27... (valid: 140,000)

  Positions: 500,000 | Valid: 140,300
  Horizon 1: 140,300 sequences
  Horizon 2: 140,300 sequences
  ...
  Horizon 10: 140,300 sequences

Processing VAL data...
Creating sequences for val...
Processing 27 containers...
  Horizon 1: 70,150 sequences
  ...

Processing TEST data...
Creating sequences for test...
Processing 27 containers...
  Horizon 1: 47,433 sequences
  ...

✓ All sequences generated successfully
```

---

### Cell 17: Verify Sequences
**What it does:** Loads and checks generated sequences

**Output you should see:**
```
======================================================================
VERIFYING SEQUENCES
======================================================================

Training Data (Horizon 1):
  X shape: (140300, 240, 34)
    - Sequences: 140,300
    - Timesteps: 240
    - Features: 34
  y shape: (140300, 4)
    - Sequences: 140,300
    - Metrics: 4

  Validation: 70,150 sequences
  Test: 47,433 sequences

  Total .npy files: 30
  Total .json files: 10

======================================================================
✅ SEQUENCES LOADED SUCCESSFULLY!
======================================================================

Ready for Phase 2: GRU Model Training
```

---

### Cell 19: Download/Upload Options
**What it does:** Provides code snippets for downloading or saving to Google Drive

**Option A: Download to Local Computer**
```python
from google.colab import files
import shutil

shutil.make_archive('/tmp/sequences', 'zip', sequences_path)
files.download('/tmp/sequences.zip')
```

**Option B: Save Back to Google Drive**
```python
import shutil
gd_sequences = Path('/content/drive/My Drive/sequences')
gd_sequences.mkdir(exist_ok=True)
for file in sequences_path.glob('*'):
    shutil.copy(file, gd_sequences / file.name)
```

---

## ⚠️ Common Issues & Fixes

### Issue 1: "raw folder not found"

**Error message:**
```
❌ raw folder not accessible at /content/drive/My Drive/raw
```

**Causes:**
- Folder doesn't exist at the right location
- Folder named something else (case-sensitive)
- Google Drive folder not shared with your account

**Fixes:**
1. Create folder: `My Drive/raw/`
2. Upload CSV files to subfolders
3. Check folder permissions
4. Re-run Cell 3 (mount again)
5. Re-run Cell 6 (verify access)

---

### Issue 2: "Permission Error when mounting"

**Error message:**
```
PermissionError: Cannot mount Google Drive
```

**Fix:**
1. Re-run Cell 3
2. Click the authorization link
3. Select your Google account
4. Click "Allow"
5. Copy the code
6. Paste in Colab input box
7. Press Enter

---

### Issue 3: "No CSV files found"

**Error message:**
```
No CSV files found!
❌ Failed to load data
```

**Causes:**
- CSV files not in correct nested folders
- Wrong folder structure (files not in `container/` subfolder)
- Wrong file extension (.csv)

**CORRECT structure (notebook searches here):**
```
My Drive/raw/
├── complex/case1/container/*.csv  ← CSV files HERE
├── complex/case2/container/*.csv  ← CSV files HERE
└── single/case2/container/*.csv   ← CSV files HERE
```

**Fixes:**
1. Verify CSV files are in `.../container/` folders (NOT in case folder directly)
2. Check that folder names match:
   - `raw/` (main folder)
   - `complex/` (case group)
   - `case1/` and `case2/` (case subfolders)
   - `container/` (data folder - THIS is where CSV files go!)
3. Verify all CSV files end with `.csv`
4. Re-upload to correct folder locations if needed

**Pro tip:** The notebook uses `raw.glob('**/*.csv')` which finds files at ANY depth, so it will find your CSV files as long as they're somewhere in the `raw/` folder hierarchy.

---

### Issue 4: "Sequences not generated"

**Error message:**
```
❌ Error: [Errno 2] No such file or directory
Sequences may not have been generated
```

**Causes:**
- Cell 15 didn't complete
- Data had too many NaN values
- Container_id column missing
- Timestamp column missing

**Fixes:**
1. Check data preview in Cell 8
2. Look for columns: `new_container_id`, `timestamp`
3. Re-run Cell 15 (full sequence generation)
4. Wait for completion (5-10 minutes)
5. Check logs for errors

---

### Issue 5: "Runtime crashed during sequences"

**Error message:**
```
Kernel died, restarting...
```

**Cause:** Out of memory (too much data loaded at once)

**Fix:** Already handled! This notebook uses batch processing (1000 sequences at a time) to avoid this issue. If it still happens:
1. Runtime → Restart
2. Run cells in order (don't skip)
3. Make sure you're using Complete_Pipeline_Google_Drive_Data.ipynb (not other variants)

---

## 📊 Expected Data Statistics

After successful completion:

| Metric | Value |
|--------|-------|
| Train sequences | ~140,300 |
| Val sequences | ~70,150 |
| Test sequences | ~47,433 |
| Timesteps per sequence | 240 |
| Features per timestep | 34 |
| Target metrics | 4 (CPU, Memory types) |
| Horizons | 10 (1-10 steps ahead) |
| Total files | 30+ (.npy + .json) |
| Total size | ~5.5 GB |
| Runtime | 5-15 minutes |

---

## 🎯 Next Steps: GRU Model Training

Once sequences are generated, load and train:

```python
from pathlib import Path
import numpy as np
import tensorflow as tf

sequences_path = Path('/content/processed_data/sequences')

# Load horizon 1 data
X_train = np.load(sequences_path / 'sequences_horizon_1_X_train.npy')
y_train = np.load(sequences_path / 'sequences_horizon_1_y_train.npy')
X_val = np.load(sequences_path / 'sequences_horizon_1_X_val.npy')
y_val = np.load(sequences_path / 'sequences_horizon_1_y_val.npy')

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.GRU(64, return_sequences=True, input_shape=(240, 34)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.GRU(32),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(4)  # 4 target metrics
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    verbose=1
)
```

---

## 🆘 Need Help?

**For issues with:**
- Google Drive access → Check folder structure section
- Data loading → Check Common Issues section
- Sequence generation → Check logs in Cell 15 output
- Google Colab → https://colab.research.google.com/help

**Project contact:** Team-Dracasys (dracasys@gmail.com)

---

## ✅ Checklist Before Running

- [ ] Google Drive has `My Drive/raw/` folder
- [ ] Subfolders exist with CSV files
- [ ] Notebook: `Complete_Pipeline_Google_Drive_Data.ipynb`
- [ ] Opened in https://colab.research.google.com
- [ ] Read this guide
- [ ] Ready to run!

---

**Total time from start to sequences ready: 15-20 minutes**

**Status: ✅ Ready for Phase 2 (GRU Training)**
