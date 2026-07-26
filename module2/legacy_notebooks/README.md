# Notebooks Folder - Drift-Aware Resource Prediction Pipeline

## Overview

This folder contains the **complete end-to-end Jupyter Notebook** for preprocessing and sequence generation.

**Perfect for:** Google Colab execution (no local installation needed!)

---

## Files in This Folder

### 1. 📓 `Complete_Pipeline_Phase0_Phase1.ipynb` (MAIN NOTEBOOK)
   - **What:** Complete pipeline from raw data to ML sequences
   - **Size:** 50 KB
   - **Duration:** 5-7 minutes runtime
   - **For:** Google Colab (optimized) or local Jupyter
   - **Includes:**
     - Phase 0: Data preprocessing (4 steps)
     - Phase 1: Sequence generation (memory-efficient streaming)
     - Data loading helpers
     - Verification utilities

---

### 2. 📖 `QUICK_START.md` (START HERE)
   - **What:** 5-minute setup guide
   - **Best for:** First-time users
   - **Covers:**
     - Step-by-step instructions (4 steps)
     - Common errors & fixes
     - Visual folder structure
     - Success checklist
   - **Read this first!** ⭐

---

### 3. 📚 `COLAB_USAGE_GUIDE.md` (DETAILED REFERENCE)
   - **What:** Complete Google Colab documentation
   - **Best for:** Troubleshooting, understanding details
   - **Covers:**
     - Cell-by-cell breakdown
     - 3 data upload options
     - Performance benchmarks
     - Local execution setup
     - Best practices
   - **Read after Quick Start**

---

## Quick Navigation

### I want to run this in Google Colab RIGHT NOW
→ Read **QUICK_START.md** (5 minutes)

### I'm having issues with Google Colab
→ See **COLAB_USAGE_GUIDE.md** Troubleshooting section

### I want to understand what each cell does
→ Read **COLAB_USAGE_GUIDE.md** Step-by-Step Execution section

### I want to run this locally (not Colab)
→ See **COLAB_USAGE_GUIDE.md** "Running on Local Machine" section

---

## The 4-Step Process

```
Step 1: Download Notebook (1 minute)
    ↓
Step 2: Setup Google Drive Folder Structure (2 minutes)
    ↓
Step 3: Upload CSV Data Files (5-10 minutes)
    ↓
Step 4: Run Notebook in Google Colab (5-7 minutes)
    ↓
✅ DONE - Sequences ready for GRU training!
```

---

## What The Notebook Does

### Input
```
Raw CSV files (container + machine metrics)
├── container_metrics_*.csv
└── machine_metrics_*.csv
```

### Processing (5-7 minutes)

**Phase 0: Preprocessing**
1. Merge metrics from multiple sources
2. Split chronologically (prevent data leakage)
3. Normalize per container
4. Engineer temporal features

**Phase 1: Sequence Generation**
5. Create 240-step lookback sequences
6. Generate 1-10 step prediction horizons
7. Save in NumPy binary format (fast, memory-efficient)

### Output
```
Sequences ready for ML (5.5 GB)
├── sequences_horizon_1_X_train.npy     (140,300 sequences)
├── sequences_horizon_1_y_train.npy     (140,300 targets)
├── sequences_horizon_1_metadata_train.json
├── ... (horizons 2-10)
├── ... (val and test datasets)
└── (30 files total)

Shapes:
  X: (n_sequences, 240_timesteps, 34_features)
  y: (n_sequences, 4_target_metrics)
```

---

## System Requirements

### For Google Colab (Recommended)
- ✅ Google Account
- ✅ Google Drive (~6 GB free space)
- ✅ Internet connection
- ✅ No software to install

### For Local Jupyter
- Python 3.8+
- pandas, numpy, scikit-learn
- ~2 GB RAM
- 6 GB storage

---

## Google Drive Folder Structure Setup

**Copy this exactly:**
```
My Drive/
└── Module2/
    └── module2/
        └── data/
            └── raw/
                ├── container_metrics_*.csv    ← YOUR FILES GO HERE
                └── machine_metrics_*.csv
```

⚠️ **Important:** Folder names are case-sensitive. Use exact names above.

---

## Key Features

✅ **Designed for Google Colab**
- Automatic Google Drive mounting
- Paths configured for Colab environment
- No additional setup needed

✅ **Memory Efficient**
- Batch streaming (1000 sequences at a time)
- Peak memory usage: 32 MB
- No "Out of Memory" errors

✅ **Data Leakage Prevention**
- Split FIRST, then normalize AFTER
- Each dataset uses only its own statistics
- Per-container grouping maintained

✅ **Comprehensive Logging**
- Progress shown for every step
- Detailed output of what's happening
- Easy to debug if issues arise

✅ **Ready for Next Phase**
- Generates sequences in NumPy format
- Perfect for TensorFlow/Keras GRU training
- Includes load helper functions

---

## Expected Runtime

| Phase | Duration |
|-------|----------|
| Setup + Data Check | 1 min |
| Merge Metrics | 30 sec |
| Split & Normalize | 1 min |
| Per-Container Norm | 1 min |
| Feature Engineering | 1-2 min |
| **Sequence Generation** | **2-3 min** |
| Verify & Load | <1 min |
| **TOTAL** | **5-7 min** |

---

## Success Looks Like This

### When you run Cell 3:
```
Using Colab path: /content/drive/My Drive/Module2
📁 Directory Structure:
  Raw data: .../data/raw
  Processed: .../data/processed
  Merged: .../data/merged
  Sequences: .../data/sequences

✓ Found 2 CSV files in raw data
```

### At the end (Cell 15):
```
======================================================================
✅ ALL SEQUENCES LOADED SUCCESSFULLY!
======================================================================

Ready for Phase 2: GRU Model Training

Training: 140,300 sequences
Validation: 70,150 sequences
Test: 47,433 sequences

Example training data:
  First timestep: [0.123 -0.456 0.789 ...] ...
  Last timestep: [0.234 -0.567 0.890 ...] ...
  Target: [0.1 0.2 0.3 0.4]
```

---

## After Running - What's Generated?

### In Google Drive:
```
/Module2/module2/data/
├── raw/                    (original input)
├── processed/              (intermediate files)
├── merged/                 (engineered features)
└── sequences/              (ML-ready data ✅)
    ├── sequences_horizon_1_X_train.npy
    ├── sequences_horizon_1_y_train.npy
    ├── sequences_horizon_1_metadata_train.json
    └── ... (30 files, 5.5 GB total)
```

### Ready for Phase 2:
```python
import numpy as np

X_train = np.load('.../sequences_horizon_1_X_train.npy')
y_train = np.load('.../sequences_horizon_1_y_train.npy')

# Now train GRU model with:
# X_train.shape = (140300, 240, 34)
# y_train.shape = (140300, 4)
```

---

## Troubleshooting Quick Links

| Problem | Solution |
|---------|----------|
| Data not found | See QUICK_START.md → "If Something Goes Wrong" |
| Colab authorization fails | See COLAB_USAGE_GUIDE.md → Problem 2 |
| Notebook runs slowly | See COLAB_USAGE_GUIDE.md → Problem 3 |
| Memory error | See COLAB_USAGE_GUIDE.md → Problem 4 |
| Import errors | See COLAB_USAGE_GUIDE.md → Problem 5 |

---

## FAQ

**Q: Can I use this notebook locally?**  
A: Yes! See COLAB_USAGE_GUIDE.md section "Running on Local Machine"

**Q: What if my data is larger?**  
A: The notebook handles batches automatically. Just upload larger files.

**Q: Can I resume if Colab times out?**  
A: Yes! All progress is saved to Google Drive. Just run again.

**Q: How do I use the generated sequences?**  
A: See end of notebook or COLAB_USAGE_GUIDE.md Phase 2 section

**Q: Do I need to modify any code?**  
A: No! Notebook works as-is for Colab. For local execution, see guide.

**Q: How do I download the sequences?**  
A: See COLAB_USAGE_GUIDE.md "Best Practices" section

---

## Reading Order

1. **First visit:** Start with `QUICK_START.md` ⭐
2. **Setup:** Follow 4-step process
3. **Running:** Execute `Complete_Pipeline_Phase0_Phase1.ipynb`
4. **Issues:** Reference `COLAB_USAGE_GUIDE.md`
5. **Next phase:** See Phase 2 documentation

---

## File Details

| File | Type | Size | Purpose |
|------|------|------|---------|
| Complete_Pipeline_Phase0_Phase1.ipynb | Jupyter | 50 KB | Main executable notebook |
| QUICK_START.md | Markdown | 8.6 KB | Fast setup guide |
| COLAB_USAGE_GUIDE.md | Markdown | 16 KB | Detailed reference |
| README.md | Markdown | This file | Navigation guide |

---

## Dependencies

The notebook uses only standard libraries:

```python
# Pre-installed in Colab/Jupyter
pandas          # Data manipulation
numpy           # Numerical computing
scikit-learn    # Machine learning utilities
pathlib         # File operations
json            # Metadata handling
logging         # Progress logging
```

**No installation needed** - already available in Google Colab!

---

## Version Info

- **Created:** May 16, 2026
- **Python:** 3.8+
- **Tested on:** Google Colab, Local Jupyter
- **Last Updated:** May 16, 2026
- **Status:** ✅ Production Ready

---

## Contact & Support

**Author:** Team-Dracasys  
**Email:** dracasys@gmail.com  
**GitHub:** [AIOpsArena](https://github.com/AIOpsArena/dataset)  

---

## Quick Reference

### Most Used Links in Guides:

**QUICK_START.md:**
- Google Drive folder setup
- Data upload instructions
- Run notebook steps
- Error fixes

**COLAB_USAGE_GUIDE.md:**
- Cell-by-cell explanation
- Data upload (3 methods)
- Troubleshooting guide
- Performance metrics
- Local execution setup

---

## Next Steps After Running

✅ **Phase 1 Complete:** Notebook finishes with sequences generated

📚 **Phase 2 Ready:** Sequences are ready for GRU model training

🚀 **Ready to Build ML Model:** Use loaded sequences with TensorFlow

---

**Start with:** `QUICK_START.md` 👈 Read first!

Then run: `Complete_Pipeline_Phase0_Phase1.ipynb` 👈 Execute in Colab

Questions: `COLAB_USAGE_GUIDE.md` 👈 Reference for details

---

**Happy machine learning! 🎉**
