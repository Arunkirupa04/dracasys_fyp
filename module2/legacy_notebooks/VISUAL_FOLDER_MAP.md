# 🗺️ Visual Folder Map - Complete Structure

## 📍 One-Click Navigation Guide

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Module2 (ROOT FOLDER)                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  📁 notebooks/ ⭐ START HERE                                        │
│  ├─ 📓 Complete_Pipeline_Phase0_Phase1.ipynb ← MAIN NOTEBOOK      │
│  ├─ 📖 README.md ← READ THIS FIRST                                │
│  ├─ 🚀 QUICK_START.md ← 5-MINUTE SETUP                           │
│  ├─ 📚 COLAB_USAGE_GUIDE.md ← DETAILED GUIDE                     │
│  └─ 📋 SUMMARY.txt ← FILE SUMMARY                                │
│                                                                      │
│  📁 module2/                                                        │
│  ├─ 📁 preprocessing/ ✅ ALL PYTHON SCRIPTS                       │
│  │  ├─ phase1_sequence_generation_ultra_fast.py                  │
│  │  ├─ feature_engineering.py                                    │
│  │  ├─ merge_cases.py                                            │
│  │  ├─ normalize_and_merge.py                                    │
│  │  └─ ... (11+ scripts total)                                   │
│  │                                                               │
│  ├─ 📁 data/ ⭐ YOUR DATA GOES HERE                              │
│  │  ├─ 📁 raw/ ← UPLOAD CSV FILES HERE                          │
│  │  │  ├─ container_metrics_*.csv                              │
│  │  │  └─ machine_metrics_*.csv                                │
│  │  │                                                          │
│  │  ├─ 📁 processed/ (auto-generated)                          │
│  │  ├─ 📁 merged/ (auto-generated)                             │
│  │  └─ 📁 sequences/ (auto-generated - 5.5 GB)                 │
│  │                                                              │
│  ├─ 📁 notebooks/ (old analysis notebooks)                      │
│  │                                                              │
│  └─ other folders...                                           │
│                                                                  │
│  📄 README.md                                                   │
│  📄 GIT_SETUP.md                                               │
│  📄 FOLDER_STRUCTURE.md                                        │
│  📄 .gitignore                                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 What Goes Where

### **FOR GOOGLE COLAB USERS**

```
Step 1: DOWNLOAD
  From: Your computer
  File: Module2/notebooks/Complete_Pipeline_Phase0_Phase1.ipynb
  
Step 2: CREATE IN GOOGLE DRIVE
  My Drive
  └── Module2
      └── module2
          └── data
              └── raw  ← CREATE THIS FOLDER
              
Step 3: UPLOAD DATA
  Upload to: My Drive/Module2/module2/data/raw/
  Files: Your CSV files
  
Step 4: OPEN IN COLAB
  Go to: colab.research.google.com
  Upload: Complete_Pipeline_Phase0_Phase1.ipynb
  
Step 5: RUN
  Click: Runtime → Run All
  Wait: 7 minutes
  Done: ✅ Sequences ready!
```

---

### **FOR LOCAL USERS**

```
Step 1: FIND NOTEBOOK
  Location: C:\Users\rasen\Documents\Claude\Projects\Module2\notebooks\
  File: Complete_Pipeline_Phase0_Phase1.ipynb
  
Step 2: PLACE DATA
  Location: C:\Users\rasen\Documents\Claude\Projects\Module2\module2\data\raw\
  Files: Your CSV files
  
Step 3: OPEN NOTEBOOK
  Tool: Jupyter Notebook / JupyterLab
  Open: Complete_Pipeline_Phase0_Phase1.ipynb
  
Step 4: RUN
  Click: Run All Cells
  Or: Run cells one-by-one
  Wait: 7 minutes
  Done: ✅ Sequences ready!
```

---

## 📊 Data Flow Diagram

```
YOUR RAW CSV FILES
        ↓
   [UPLOAD]
        ↓
┌─────────────────────┐
│  raw/ (YOUR INPUT)  │
├─────────────────────┤
│ container_metrics   │
│ machine_metrics     │
└─────────────────────┘
        ↓
   [Phase 0.1]
   MERGE
        ↓
┌──────────────────────┐
│ processed/ (GEN)     │
├──────────────────────┤
│ merged_metrics.csv   │
└──────────────────────┘
        ↓
   [Phase 0.2-0.4]
   NORMALIZE + ENGINEER
        ↓
┌──────────────────────┐
│ merged/ (GEN)        │
├──────────────────────┤
│ train_with_features  │
│ val_with_features    │
│ test_with_features   │
└──────────────────────┘
        ↓
   [Phase 1]
   SEQUENCES
        ↓
┌──────────────────────┐
│ sequences/ (GEN)     │
├──────────────────────┤
│ *.npy files (30)     │
│ 5.5 GB total         │
│ Ready for ML! ✅     │
└──────────────────────┘
```

---

## 🎨 Folder Size Reference

```
Before Running:
├── notebooks/          ~100 KB
├── preprocessing/      ~200 KB
├── data/raw/           YOUR FILES SIZE
└── Total:              Minimal

After Running (Complete Pipeline):
├── notebooks/          ~100 KB
├── preprocessing/      ~200 KB
├── data/
│  ├── raw/             YOUR FILES SIZE
│  ├── processed/       ~2.0 GB
│  ├── merged/          ~1.5 GB
│  ├── sequences/       ~5.5 GB ⭐
│  └── Total:           ~9 GB
└── Total with code:    ~9+ GB
```

---

## 🔐 Path References for Copy-Paste

### **Google Colab - In Notebook:**
```python
# Notebook uses automatically:
base_path = Path('/content/drive/My Drive/Module2')
data_path = base_path / 'module2' / 'data'
raw_data_path = data_path / 'raw'
```

### **Google Drive - Create:**
```
My Drive/Module2/module2/data/raw/
```

### **Windows Local:**
```
C:\Users\rasen\Documents\Claude\Projects\Module2\module2\data\raw\
```

### **Mac/Linux Local:**
```
~/Documents/Claude/Projects/Module2/module2/data/raw/
```

---

## ✅ File Location Checklist

### **Before Starting:**

```
✓ NOTEBOOKS FOLDER
  └─ Module2/notebooks/
     ├─ Complete_Pipeline_Phase0_Phase1.ipynb    50 KB
     ├─ README.md                                9 KB
     ├─ QUICK_START.md                           9 KB
     ├─ COLAB_USAGE_GUIDE.md                    16 KB
     └─ SUMMARY.txt                             14 KB

✓ DATA STRUCTURE
  └─ Module2/module2/data/
     ├─ raw/          ← UPLOAD YOUR CSV FILES
     ├─ processed/    (empty, will be filled)
     ├─ merged/       (empty, will be filled)
     └─ sequences/    (empty, will be filled)

✓ PYTHON SCRIPTS
  └─ Module2/module2/preprocessing/
     ├─ phase1_sequence_generation_ultra_fast.py
     ├─ feature_engineering.py
     ├─ ... (11+ scripts)
     └─ All still available for command-line use
```

### **After Running:**

```
✓ DATA GENERATED
  └─ Module2/module2/data/
     ├─ processed/
     │  ├─ merged_metrics.csv
     │  ├─ train/val/test_data_normalized.csv
     │  └─ train/val/test_data_per_container_normalized.csv
     │
     ├─ merged/
     │  ├─ train_data_with_features.csv
     │  ├─ val_data_with_features.csv
     │  └─ test_data_with_features.csv
     │
     └─ sequences/
        ├─ 30 .npy files
        ├─ 30 .json metadata files
        └─ 5.5 GB total ✅
```

---

## 🚨 Common Path Mistakes & Corrections

### ❌ **WRONG:** Upload to root of Google Drive
```
My Drive/
├─ container_metrics.csv ❌ WRONG LOCATION
└─ machine_metrics.csv ❌
```

### ✅ **RIGHT:** Upload to correct subfolder
```
My Drive/
└─ Module2/
   └─ module2/
      └─ data/
         └─ raw/
            ├─ container_metrics.csv ✅ CORRECT
            └─ machine_metrics.csv ✅
```

---

### ❌ **WRONG:** CSV files in data/ root
```
data/
├─ container_metrics.csv ❌
├─ machine_metrics.csv ❌
├─ raw/
├─ processed/
└─ sequences/
```

### ✅ **RIGHT:** CSV files in raw/ subfolder
```
data/
├─ raw/
│  ├─ container_metrics.csv ✅
│  ├─ machine_metrics.csv ✅
│  ├─ processed/
│  ├─ merged/
│  └─ sequences/
```

---

## 📍 GPS Coordinates (File Paths)

### **Notebook Location**
```
🗺️  C:\Users\rasen\Documents\Claude\Projects\Module2\notebooks\
📌 Complete_Pipeline_Phase0_Phase1.ipynb
```

### **Raw Data Location (Local)**
```
🗺️  C:\Users\rasen\Documents\Claude\Projects\Module2\module2\data\raw\
📌 container_metrics_*.csv
📌 machine_metrics_*.csv
```

### **Raw Data Location (Google Drive)**
```
🗺️  Google Drive > My Drive > Module2 > module2 > data > raw
📌 container_metrics_*.csv
📌 machine_metrics_*.csv
```

### **Output Location**
```
🗺️  C:\Users\rasen\Documents\Claude\Projects\Module2\module2\data\sequences\
📌 sequences_horizon_*.npy (30 files)
📌 sequences_horizon_*_metadata_*.json (30 files)
```

---

## 🎓 Understanding the Notebook Path Logic

### **The notebook automatically detects your environment:**

```
if IN_COLAB:
    # You're in Google Colab
    base_path = '/content/drive/My Drive/Module2'
    # Notebook uses Google Drive paths
else:
    # You're running locally
    base_path = 'C:/Users/rasen/Documents/Claude/Projects/Module2'
    # Notebook uses local paths

# Both automatically create missing directories:
data_path.mkdir(parents=True, exist_ok=True)
```

**Result:** Same notebook works everywhere! ✅

---

## 🔄 How Paths Flow Through Notebook

```
CELL 3: Setup Paths
    ↓
CELLS 4-5: Load & Verify
    ↓
CELLS 6-8: Merge (raw/ → processed/)
    ↓
CELLS 9-10: Split (processed/ → processed/)
    ↓
CELLS 11-12: Per-Container (processed/ → processed/)
    ↓
CELLS 13-14: Features (processed/ → merged/)
    ↓
CELLS 15-16: Sequences (merged/ → sequences/)
    ↓
CELLS 17-18: Verify (sequences/ → output display)
    ↓
✅ DONE - All 30 files in sequences/
```

---

## 📚 Quick Reference Card

| What | Where | What It Contains |
|------|-------|-----------------|
| **Notebook** | `Module2/notebooks/` | Complete pipeline code |
| **Read First** | `notebooks/README.md` | Navigation guide |
| **Setup Guide** | `notebooks/QUICK_START.md` | 5-min instructions |
| **Detailed Help** | `notebooks/COLAB_USAGE_GUIDE.md` | Complete reference |
| **Raw Data IN** | `module2/data/raw/` | Your CSV files |
| **Scripts** | `module2/preprocessing/` | Python code (still there) |
| **Output** | `module2/data/sequences/` | ML-ready sequences |

---

## 🎯 Key Takeaways

✅ **Notebook:** `Module2/notebooks/Complete_Pipeline_Phase0_Phase1.ipynb`

✅ **Data Input:** `Module2/module2/data/raw/` (local) or `My Drive/Module2/module2/data/raw/` (Colab)

✅ **All Paths Auto-Configured** - No manual path changes needed

✅ **Works Local + Colab** - Same notebook, both environments

✅ **Creates Missing Folders** - Don't manually create processed/, merged/, sequences/

✅ **Python Scripts Preserved** - Still in preprocessing/ folder

✅ **Documentation Complete** - Every possible guide provided

---

## 🚀 Ready to Start?

1. **Download notebook** from `Module2/notebooks/`
2. **Read** `notebooks/README.md` (2 min)
3. **Follow** `notebooks/QUICK_START.md` (5 min)
4. **Run** notebook in Colab or Jupyter (7 min)
5. **Done!** ✅

**Total: 15 minutes to ML-ready data** 🎉

---

**All paths verified. All systems go.** 🚀
