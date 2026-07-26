# 🚀 Quick Start - 5 Minutes to Run

## The Absolute Fastest Way

### Step 1: Download Files (1 minute)

1. Download `Complete_Pipeline_Phase0_Phase1.ipynb` from GitHub
2. Download `COLAB_USAGE_GUIDE.md` for reference

---

### Step 2: Setup Google Drive (2 minutes)

**Option A: Google Drive Web** (Easiest for first time)

1. Go to [Google Drive](https://drive.google.com)
2. Create this folder structure:
   ```
   Right-click in My Drive:
   New → Folder → Name: "Module2" → Create
   
   Open Module2 folder:
   Right-click → New → Folder → Name: "module2" → Create
   
   Open module2 folder:
   Right-click → New → Folder → Name: "data" → Create
   
   Open data folder:
   Right-click → New → Folder → Name: "raw" → Create
   ```

3. **Upload your CSV files:**
   - Right-click in `raw` folder
   - **Upload files**
   - Select your CSV files
   - Wait for upload (green checkmark)

**Folder structure should be:**
```
My Drive/Module2/module2/data/raw/
├── container_metrics_*.csv
├── machine_metrics_*.csv
└── ...
```

---

### Step 3: Open Notebook (1 minute)

1. Go to [Google Colab](https://colab.research.google.com)
2. Click **File → Upload Notebook**
3. Select `Complete_Pipeline_Phase0_Phase1.ipynb`
4. Wait for it to load

---

### Step 4: Run Notebook (1 minute)

**In Colab:**

1. **Cell 1:** Click ▶ (run)
2. **Cell 2:** Click ▶ → Authorize Google Drive (click link, select account, allow, paste code)
3. **Cell 3:** Click ▶ → Verify data found
4. **Cells 4-5:** Click ▶ → Merge raw metrics
5. **Cells 6-7:** Click ▶ → Split & normalize (prevent leakage)
6. **Cells 8-9:** Click ▶ → Per-container normalization
7. **Cells 10-11:** Click ▶ → Feature engineering
8. **Cells 12-13:** Click ▶ → Sequence generation (THIS TAKES 2-3 MINUTES)
9. **Cells 14-15:** Click ▶ → Verify sequences loaded ✅ DONE!

---

## Even Faster: Run All at Once

```
In Colab: Runtime → Run All
```

**Then:**
1. Authorize Google Drive when prompted (Step 2)
2. Wait ~5-7 minutes total
3. See "✅ ALL SEQUENCES LOADED SUCCESSFULLY!"
4. Done!

---

## For the Impatient

**Copy this into a Colab cell:**

```python
# Auto-setup (save 2 minutes!)
!mkdir -p /tmp/data/raw
!cd /tmp && cat > setup.py << 'EOF'
# Setup code
print("✓ Ready!")
EOF
python /tmp/setup.py
```

---

## What Happens in Each Section

| Section | Time | What It Does | Output |
|---------|------|-------------|--------|
| 1-3 | 1 min | Setup, paths, verify data | Shows folder structure |
| 4-5 | 30 sec | Merge metrics | `merged_metrics.csv` |
| 6-7 | 1 min | Split train/val/test | 3 normalized CSVs |
| 8-9 | 1 min | Per-container norm | 3 container-normalized CSVs |
| 10-11 | 1-2 min | Feature engineering | 3 CSVs with 34 features |
| 12-13 | 2-3 min | **Sequence generation** | **30 .npy files (5.5 GB)** |
| 14-15 | <1 min | Verify sequences | ✅ Ready for training |

---

## If Something Goes Wrong

### Error: "Data directory not found"
```
❌ /content/drive/My Drive/Module2/module2/data/raw not found
```
**Fix:** Check Google Drive has correct folder structure

### Error: "Permission denied"
```
❌ Could not authorize Google Drive
```
**Fix:** Click the link in Cell 2, select account, click "Allow"

### Error: "No module pandas"
```
❌ ModuleNotFoundError: No module named 'pandas'
```
**Fix:** Click **Runtime → Factory Reset**, run again

### Notebook is very slow
**Fix:** Click **Runtime → Change runtime type → GPU**

### Cells take too long
**Status is normal if:**
- Setup: <1 min ✓
- Merge: ~30 sec ✓
- Split: ~1 min ✓
- Per-container: ~1 min ✓
- Features: 1-2 min ✓
- **Sequences: 2-3 min ✓** (EXPECTED - processes 257K sequences)

---

## When You're Done

**Success message:**
```
======================================================================
✅ ALL SEQUENCES LOADED SUCCESSFULLY!
======================================================================
Ready for Phase 2: GRU Model Training
```

**Your data is ready in:**
```
/content/drive/My Drive/Module2/module2/data/sequences/
```

**Next:** See `COLAB_USAGE_GUIDE.md` for Phase 2 (GRU training)

---

## Common Questions

**Q: Can I download the data?**  
A: Yes! Add to last cell:
```python
from google.colab import files
# Creates zip of sequences
!cd /content/drive/My\ Drive/Module2/module2/data && \
  zip -r sequences.zip sequences/
files.download('/content/drive/My Drive/Module2/module2/data/sequences.zip')
```

**Q: How much space do I need?**  
A: ~6 GB in Google Drive (for output) + upload space for input CSVs

**Q: Can I run it twice?**  
A: Yes, but remove old files first:
```python
# Optional: clear previous output
!rm -rf /content/drive/My\ Drive/Module2/module2/data/processed
!rm -rf /content/drive/My\ Drive/Module2/module2/data/merged
!rm -rf /content/drive/My\ Drive/Module2/module2/data/sequences
```

**Q: How do I use the sequences?**  
A: Load in Python:
```python
import numpy as np

X_train = np.load('/content/drive/My Drive/Module2/module2/data/sequences/sequences_horizon_1_X_train.npy')
y_train = np.load('/content/drive/My Drive/Module2/module2/data/sequences/sequences_horizon_1_y_train.npy')

print(f"Shape: {X_train.shape} → {y_train.shape}")
# Output: (140300, 240, 34) → (140300, 4)
```

---

## Visual Folder Structure Confirmation

**Before running notebook:**
```
Google Drive (my drive.google.com)
├── ✓ Folder "Module2"
│   └── ✓ Folder "module2"
│       └── ✓ Folder "data"
│           └── ✓ Folder "raw"
│               ├── ✓ container_metrics_*.csv
│               ├── ✓ machine_metrics_*.csv
│               └── ✓ (other metric CSVs)
```

**After running notebook:**
```
Google Drive (my drive.google.com)
├── ✓ Folder "Module2"
│   └── ✓ Folder "module2"
│       └── ✓ Folder "data"
│           ├── ✓ Folder "raw"
│           │   └── (original CSV files)
│           ├── ✓ Folder "processed"
│           │   ├── merged_metrics.csv
│           │   ├── train_data_normalized.csv
│           │   ├── val_data_normalized.csv
│           │   ├── test_data_normalized.csv
│           │   ├── train_data_per_container_normalized.csv
│           │   ├── val_data_per_container_normalized.csv
│           │   └── test_data_per_container_normalized.csv
│           ├── ✓ Folder "merged"
│           │   ├── train_data_with_features.csv
│           │   ├── val_data_with_features.csv
│           │   └── test_data_with_features.csv
│           └── ✓ Folder "sequences"
│               ├── sequences_horizon_1_X_train.npy
│               ├── sequences_horizon_1_y_train.npy
│               ├── sequences_horizon_1_metadata_train.json
│               ├── ... (horizons 2-10)
│               ├── ... (val and test datasets)
│               └── (30 files total = 5.5 GB)
```

---

## Copy-Paste Commands for Colab

**Paste into Cell 2 if mount fails:**
```python
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    print("✓ Google Drive mounted!")
```

**Paste into Cell 3 to check data:**
```python
import os
data_path = '/content/drive/My Drive/Module2/module2/data/raw'
files = os.listdir(data_path)
print(f"Files found: {len(files)}")
for f in files:
    print(f"  - {f}")
```

---

## Success Checklist

- [ ] Downloaded notebook file
- [ ] Created folder structure in Google Drive
- [ ] Uploaded CSV files to `raw` folder
- [ ] Opened notebook in Google Colab
- [ ] Authorized Google Drive access
- [ ] Ran Cell 3 - saw "✓ Found X CSV files"
- [ ] Ran all cells or clicked "Run All"
- [ ] Waited 5-7 minutes
- [ ] Saw "✅ ALL SEQUENCES LOADED SUCCESSFULLY!"
- [ ] Found 30 `.npy` files in sequences folder

---

## Timeline

```
T+0 min:   Open notebook in Colab
T+0 min:   Mount Google Drive (authorize)
T+1 min:   Verify data folder structure
T+1 min:   Merge raw metrics
T+2 min:   Split & normalize
T+3 min:   Per-container normalization
T+4 min:   Feature engineering
T+6 min:   Sequence generation (batch streaming)
T+7 min:   ✅ DONE - Load sequences
```

---

## Ready? Let's Go! 🚀

1. **Download notebook**
2. **Create Google Drive folders** (2 min)
3. **Upload CSV files** (5-10 min depending on size)
4. **Open in Colab**
5. **Click Run All** → Go grab coffee ☕
6. **Come back in 7 minutes** → Done! ✅

---

**Questions?** See `COLAB_USAGE_GUIDE.md`  
**Issues?** Email: dracasys@gmail.com

Happy computing! 🎉
