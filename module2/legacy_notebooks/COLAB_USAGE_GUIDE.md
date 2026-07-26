# Google Colab Usage Guide - Complete Pipeline Notebook

## Overview

This guide explains how to use the **`Complete_Pipeline_Phase0_Phase1.ipynb`** notebook in Google Colab.

The notebook is a complete end-to-end pipeline that:
- Takes raw container metrics (CSV files)
- Applies preprocessing and feature engineering
- Generates machine learning sequences
- Outputs data ready for GRU model training

**Total Runtime:** 5-7 minutes (all steps combined)

---

## Quick Start (5 Minutes)

### Step 1: Open Notebook in Google Colab

1. Go to [Google Colab](https://colab.research.google.com)
2. Click **File → Open Notebook**
3. Go to **GitHub** tab
4. Enter your repository URL (if publicly available)
5. Or **Upload** the `.ipynb` file directly:
   - Download `Complete_Pipeline_Phase0_Phase1.ipynb` from your computer
   - Click **File → Upload Notebook**
   - Select the file

---

### Step 2: Prepare Data Folder in Google Drive

#### Option A: Use Google Drive Desktop Sync (Easiest)

1. **Install** [Google Drive for Desktop](https://www.google.com/drive/download/)
2. **Sign in** with your Google account
3. **Create folder structure:**
   ```
   Google Drive (local sync folder)
   └── My Drive/
       └── Module2/
           └── module2/
               └── data/
                   └── raw/
   ```
4. **Add CSV files to `raw` folder:**
   - `container_metrics_*.csv`
   - `machine_metrics_*.csv`
5. **Files auto-sync** to Google Drive cloud
6. **Colab sees them immediately**

#### Option B: Manual Upload via Google Drive Web

1. Go to [Google Drive](https://drive.google.com)
2. Create folders: `My Drive → Module2 → module2 → data → raw`
3. Right-click in `raw` folder → **Upload files**
4. Select your CSV files from your computer
5. Upload completes (~5-10 minutes for large files)

#### Option C: Upload via Colab (For Small Files Only)

```python
# In notebook cell:
from google.colab import files
uploaded = files.upload()

# Move to correct location
import shutil
import os
for filename in uploaded.keys():
    shutil.move(filename, 
        f'/content/drive/My Drive/Module2/module2/data/raw/{filename}')
    print(f"Moved {filename}")
```

---

## Step-by-Step Execution Guide

### Cell 1: Check Colab Environment
```python
# ✓ This cell checks if you're in Google Colab
# Output should show: "Running in Google Colab"
```

**What it does:** Detects Google Colab environment
**Error possible:** None - just informational

---

### Cell 2: Mount Google Drive
```python
# CRITICAL CELL - Must run this first!
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
```

**What it does:** Connects Google Drive to Colab  
**What you'll see:** A link to authorize access  
**Action required:** Click link → Select Google account → Click "Allow" → Copy code → Paste in prompt

**After running:** You should see "✓ Google Drive mounted!"

---

### Cell 3: Setup Paths
```python
# Verifies your data folder structure
# Output shows:
# - Raw data path
# - Processed path
# - Merged path
# - Sequences path
# - Number of CSV files found
```

**Expected output:**
```
Using Colab path: /content/drive/My Drive/Module2
📁 Directory Structure:
  Raw data: /content/drive/My Drive/Module2/module2/data/raw
  Processed: /content/drive/My Drive/Module2/module2/data/processed
  Merged: /content/drive/My Drive/Module2/module2/data/merged
  Sequences: /content/drive/My Drive/Module2/module2/data/sequences

✓ Found 2 CSV files in raw data
```

**If you see:** "⚠ Raw data path not found"  
**Action:** Check that you uploaded files to correct Google Drive location

---

### Cells 4-5: Phase 0 - Step 1: Merge Metrics
```python
# Loads container_metrics and machine_metrics
# Merges them on timestamp and machine_id
# Saves merged_metrics.csv
```

**Expected output:**
```
✓ Found 2 container file(s) and 2 machine file(s)
Container metrics: (140670, 38)
Machine metrics: (100000, 25)
Merged shape: (140670, 40)
✓ Saved: .../processed/merged_metrics.csv
```

**Time:** ~30 seconds

---

### Cells 6-7: Phase 0 - Step 2: Split & Normalize
```python
# Splits into train/val/test chronologically
# Normalizes each split independently (PREVENTS DATA LEAKAGE)
```

**Expected output:**
```
Splitting by timestamp (FIRST - prevent leakage)...
train: 84,402 rows (60.0%)
val: 28,134 rows (20.0%)
test: 28,134 rows (20.0%)

Normalizing train using only train statistics...
Normalizing val using only val statistics...
Normalizing test using only test statistics...

✓ Saved: train_data_normalized.csv
✓ Saved: val_data_normalized.csv
✓ Saved: test_data_normalized.csv
```

**Time:** ~1 minute

---

### Cells 8-9: Phase 0 - Step 3: Per-Container Normalization
```python
# Applies normalization per container
# Accounts for different resource usage patterns
```

**Expected output:**
```
Processing train...
Computing per-container statistics for train...
Computed stats for 27 containers
✓ Saved: train_data_per_container_normalized.csv

Processing val...
Processing test...
```

**Time:** ~1 minute

---

### Cells 10-11: Phase 0 - Step 4: Feature Engineering
```python
# Creates lag features (DIFF_1, DIFF_2, DIFF_3)
# Creates rolling statistics (MEAN_3, STD_3)
# From 4 target columns → 34 total features
```

**Expected output:**
```
--- Processing train ---
Input: 84,402 rows × 40 columns
Engineering features...
✓ Created 20 new features
Total columns: 60
Output: 84,402 rows × 60 columns

--- Processing val ---
--- Processing test ---

✓ Step 4 Complete: Features engineered
```

**Time:** ~1-2 minutes

---

### Cells 12-13: Phase 1 - Sequence Generation
```python
# Creates 240-step lookback sequences
# Generates 1-10 step horizon targets
# Saves in NumPy format (memory-efficient)
```

**Expected output:**
```
######################################################################
PHASE 1: SEQUENCE GENERATION - ULTRA FAST (Memory-Efficient)
######################################################################
Method: Streaming to disk (batch size: 1000)
Precision: float32 (50% memory savings)

======================================================================
Processing: train_data_with_features.csv
======================================================================
Loaded 84,402 rows × 60 columns
Creating sequences for train...
Processing 27 containers...
  Container 5/27... (valid: 35,200)
  ...
  Positions: 84,402 | Valid: 140,300

  Horizon 1: 140,300 sequences saved
  Horizon 2: 140,300 sequences saved
  ... (horizons 3-10)

✓ TRAIN processed successfully
✓ VAL processed successfully
✓ TEST processed successfully

Total: 3/3 datasets processed
```

**Time:** ~2-3 minutes

**Memory usage:** Peak ~32 MB (thanks to batch streaming)

---

### Cells 14-15: Load & Verify Sequences
```python
# Loads sequences to verify they're correct
# Shows shapes and sample data
```

**Expected output:**
```
======================================================================
VERIFYING GENERATED SEQUENCES
======================================================================

Loading train sequences (horizon 1)...
  ✓ Loaded 140,300 sequences
  X shape: (140300, 240, 34)
  y shape: (140300, 4)

Training: 140,300 sequences
Validation: 70,150 sequences
Test: 47,433 sequences

======================================================================
✅ ALL SEQUENCES LOADED SUCCESSFULLY!
======================================================================

Ready for Phase 2: GRU Model Training

Example training data:
  First timestep: [0.123 -0.456 0.789 ...] ...
  Last timestep: [0.234 -0.567 0.890 ...] ...
  Target: [0.1 0.2 0.3 0.4]
```

**Time:** <30 seconds

---

## Troubleshooting Guide

### Problem 1: "Data directory not found"

**Cause:** Files not uploaded to correct location

**Check:**
1. In Colab, click **Files** icon (left sidebar)
2. Click **Drive** folder
3. Navigate to: `drive/My Drive/Module2/module2/data/raw/`
4. You should see CSV files

**Fix:**
1. Go to [Google Drive](https://drive.google.com)
2. Create folders: `Module2/module2/data/raw/`
3. Upload your CSV files there
4. Wait for upload to complete
5. Run Cell 3 again

---

### Problem 2: "Permission denied" when mounting Google Drive

**Cause:** Didn't authorize Colab access

**Fix:**
1. Re-run Cell 2
2. Click the link that appears
3. Select your Google account
4. Click "Allow"
5. Copy the authorization code
6. Paste it in the Colab prompt
7. Press Enter

---

### Problem 3: Notebook runs slowly / gets stuck

**Cause:** Free Colab might have resource limits

**Solutions:**
- Use **Colab Pro** for faster GPU
- Click **Runtime → Run All** (instead of running cells one-by-one)
- Or close other Colab tabs to free resources
- Check runtime: **Runtime → Change runtime type** → Select "GPU" (free option may be available)

---

### Problem 4: "Memory error" during sequence generation

**This shouldn't happen** with the ultra-fast version, but if it does:

**Fix:**
1. Click **Runtime → Factory Reset**
2. Run all cells again from the top
3. If still fails: Your raw data might be too large
   - Reduce number of containers in preprocessing

---

### Problem 5: "ModuleNotFoundError: No module named 'pandas'"

**This shouldn't happen** - Colab has pandas pre-installed

**Fix:**
1. Click **Runtime → Factory Reset**
2. Run all cells again

---

## Google Drive Folder Structure

After everything runs, your Google Drive should look like:

```
My Drive/
└── Module2/
    └── module2/
        └── data/
            ├── raw/
            │   ├── container_metrics_*.csv  (YOUR INPUT)
            │   └── machine_metrics_*.csv
            │
            ├── processed/
            │   ├── merged_metrics.csv
            │   ├── train_data_normalized.csv
            │   ├── val_data_normalized.csv
            │   ├── test_data_normalized.csv
            │   ├── train_data_per_container_normalized.csv
            │   ├── val_data_per_container_normalized.csv
            │   └── test_data_per_container_normalized.csv
            │
            ├── merged/
            │   ├── train_data_with_features.csv
            │   ├── val_data_with_features.csv
            │   └── test_data_with_features.csv
            │
            └── sequences/
                ├── sequences_horizon_1_X_train.npy
                ├── sequences_horizon_1_y_train.npy
                ├── sequences_horizon_1_metadata_train.json
                ├── ... (horizons 2-10)
                ├── ... (val and test datasets)
                └── (30 files total, ~5-6 GB)
```

---

## Best Practices for Google Colab

### 1. **Always Mount Google Drive First**
```python
# Do this in second cell before anything else
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
```

### 2. **Use Google Drive for Persistent Storage**
- Files in `/content/` get deleted when session ends
- Files in `/content/drive/` persist permanently
- Keep your data in Google Drive

### 3. **Save Checkpoints**
```python
# After Phase 0, before Phase 1:
print("✓ Phase 0 complete - data saved to Google Drive")

# After Phase 1:
print("✓ Phase 1 complete - sequences saved to Google Drive")
```

### 4. **Monitor Memory Usage**
- Click **Runtime** → **View resources**
- Watch RAM and Disk usage
- Ultra-fast version keeps peak <50 MB

### 5. **Download Results**
```python
# Optional: Download sequences to local computer
from google.colab import files
import zipfile
import os

# Create zip
os.system('cd /content/drive/My\ Drive/Module2/module2/data && zip -r sequences.zip sequences/')

# Download
files.download('/content/drive/My Drive/Module2/module2/data/sequences.zip')
```

---

## Running on Local Machine (Optional)

If you want to run this notebook locally instead:

1. **Install Jupyter:**
   ```bash
   pip install jupyter notebook
   ```

2. **Download notebook** from GitHub

3. **Place raw data in:**
   ```
   ./module2/data/raw/
   ```

4. **Run notebook:**
   ```bash
   jupyter notebook Complete_Pipeline_Phase0_Phase1.ipynb
   ```

5. **Change path in Cell 3:**
   ```python
   # Change from:
   if IN_COLAB: base_path = Path('/content/drive/My Drive/Module2')
   # To:
   base_path = Path('.').resolve().parent
   ```

---

## Performance Metrics

### Expected Runtime

| Phase | Step | Time | Memory |
|-------|------|------|--------|
| Setup | 1-3 | <1 min | <50 MB |
| Phase 0 | Merge | ~30 sec | 500 MB |
| Phase 0 | Split & Norm | ~1 min | 800 MB |
| Phase 0 | Per-Container | ~1 min | 600 MB |
| Phase 0 | Features | ~1-2 min | 1.5 GB |
| Phase 1 | Sequences | ~2-3 min | 32 MB (batch) |
| Verify | Load & Check | <1 min | 2 GB |
| **Total** | **All Steps** | **5-7 min** | **2 GB** |

### Output Sizes

| Dataset | Size |
|---------|------|
| Training sequences | 3.0 GB |
| Validation sequences | 1.5 GB |
| Test sequences | 1.0 GB |
| **Total** | **5.5 GB** |

---

## Next Steps: Phase 2 - GRU Training

After completing this notebook, you can:

1. **Create new Colab notebook** with GRU training code
2. **Load sequences:**
   ```python
   from pathlib import Path
   import numpy as np
   import json
   
   sequences_path = Path('/content/drive/My Drive/Module2/module2/data/sequences')
   
   X_train = np.load(sequences_path / 'sequences_horizon_1_X_train.npy')
   y_train = np.load(sequences_path / 'sequences_horizon_1_y_train.npy')
   X_val = np.load(sequences_path / 'sequences_horizon_1_X_val.npy')
   y_val = np.load(sequences_path / 'sequences_horizon_1_y_val.npy')
   ```

3. **Build and train GRU model:**
   ```python
   import tensorflow as tf
   
   model = tf.keras.Sequential([
       tf.keras.layers.GRU(64, return_sequences=True, input_shape=(240, 34)),
       tf.keras.layers.Dropout(0.2),
       tf.keras.layers.GRU(32),
       tf.keras.layers.Dropout(0.2),
       tf.keras.layers.Dense(16, activation='relu'),
       tf.keras.layers.Dense(4)
   ])
   
   model.compile(optimizer='adam', loss='mse')
   model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50)
   ```

---

## FAQ

**Q: Can I run this on GPU?**  
A: Yes! Go to **Runtime → Change runtime type → GPU**. Training will be faster.

**Q: How long does data upload take?**  
A: ~5-10 minutes per GB via Google Drive web. Use Drive Desktop for faster sync.

**Q: Can I stop and resume?**  
A: Yes! After Phase 0 completes, all processed files are in Google Drive. You can resume Phase 1 anytime.

**Q: What if Colab session times out?**  
A: All progress is saved to Google Drive. Run Cell 2-3 again, then continue from where you left off.

**Q: Can multiple people use same Google Drive folder?**  
A: Yes! Share the `Module2` folder in Google Drive. Each person runs the notebook independently.

**Q: How do I convert this to run on multiple GPUs?**  
A: This notebook is designed for single GPU. For distributed training, see Phase 2 documentation.

---

## Support & Issues

**For issues:**
1. Check this troubleshooting guide first
2. Review Colab output logs
3. Restart runtime (Runtime → Restart all)
4. Email: dracasys@gmail.com

**For feature requests:**
- Create GitHub issue
- Describe what you need
- Provide error messages

---

## Summary

✅ **Notebook ready for Google Colab**  
✅ **No code changes needed**  
✅ **Google Drive integration built-in**  
✅ **Memory-efficient processing**  
✅ **Complete documentation included**  
✅ **Phase 2 ready after completion**  

---

**Happy computing! 🚀**

