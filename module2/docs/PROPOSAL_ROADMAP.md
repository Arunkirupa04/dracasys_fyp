# Module 2 Roadmap: Based on Original Proposal

## Where We Are Now

### ✅ Completed
1. **Data Acquisition & Merging** (DONE)
   - Loaded 4 cases from AIOpsArena
   - Merged into single continuous dataset
   - 533,338 rows × 11 columns
   - 27 unique containers

2. **Normalization Strategy** (DONE)
   - Per-case z-score normalization ✓
   - Per-container normalization designed (ready to implement)
   - Timestamp continuity verified (15-second intervals)
   - All 7 metrics verified & analyzed

3. **Data Quality Verification** (DONE)
   - No null values ✓
   - CPU variance confirmed ✓
   - Distribution analysis complete ✓
   - Container volatility measured (8.4x range) ✓

---

## ⚠ Implementation Status Update (Phases 1-3, post-fix)

> Added after Phases 1-3 were implemented, reviewed, and fixed in
> `final_notebook/notebook/Phase1_and_Phase2_Pipeline.ipynb`. The "Completed" section
> above is kept as historical record from an earlier exploratory pass, not edited in
> place — the reconciliation below is what the current pipeline code actually does,
> verified by executing it end-to-end (Phase 1 has no torch dependency, so this was
> fully executable outside Colab on synthetic data; Phase 2/3 model code still needs an
> actual Colab run against real data to verify).

**Phase-numbering note, read this first:** the notebook's own "Phase 1 / Phase 2 /
Phase 3" (Data Preprocessing → GRU Architecture → Training Loop) is a **different
numbering** than this roadmap document's "Phase 1-5" (Sequence Generation → GRU Model →
Adaptive Sliding Window → Drift Detection → Incremental Learning) below. The notebook's
"Phase 3" (training, now working) is not the same thing as this roadmap's "Phase 3"
(adaptive sliding window, still 100% unimplemented) — don't conflate the two when
reading status elsewhere in this file.

### Reconciled data numbers
- This section's original figure (above): 533,338 rows × 11 columns — from an earlier,
  separate exploratory pass, before the current pipeline's 7-metric KPI filter existed.
- Current pipeline's actual load (`load_data_from_drive_optimized`, 109 raw CSVs found
  at last run): **301,617 rows × 10 columns**, 27 containers.
- The ~43% gap between these two numbers is not yet explained. The pipeline now logs a
  per-case, pre/post-KPI-filter row-count breakdown on every run (added specifically for
  this) — check that log output against the 533,338 figure next time it's run in Colab
  before treating either number as authoritative.

### Split strategy — now matches this roadmap's own Option A recommendation
- Before this fix pass: a global chronological 60/20/20 cut that ignored case
  boundaries entirely.
- Now: **case-based split**, exactly as recommended in Phase 1 below — train =
  `complex_case2` + `single_case2`, val = `single_case1`, test = `complex_case1`.
  Verified by executing the actual split code against synthetic case-tagged data: the
  three splits partition cleanly by `case_source` with zero cross-contamination.
- Required adding `case_source` tracking through the raw-CSV load and pivot steps,
  which did not exist before — case identity is inferred from each raw CSV's filename.

### Architecture decision — 10 independent single-horizon models, not the single
### multi-output model this roadmap's Phase 2 recommends
- This roadmap's Phase 2 below recommends **Option A: one GRU, output all 12 steps at
  once**.
- The actual implementation trains **10 independent single-horizon `GRUModel`
  instances** (one per horizon, 1-10), each predicting 4 targets — not the single
  shared model, and not all 7 metrics / 12 horizons originally scoped.
- Rationale (decided during the Phase 1-3 fix pass, not part of the original proposal):
  lower-risk, smaller change to get one clean, verifiable training run working first,
  given the codebase was already built around per-horizon models. Migrating to a single
  multi-output model remains a valid future improvement — it would remove the current
  10x redundant training compute over very similar data.

### Other deviations from this roadmap, for the record

| Roadmap says | Actual implementation | Why |
|---|---|---|
| 7 metrics (CPU + memory + network + disk), all predicted | 7 raw metrics collected, but only **4** used as prediction targets (cpu_usage, mem_usage, mem_working_set, mem_rss) | Network/disk metrics aren't present in the AIOpsArena raw data being used; the other 3 CPU/memory metrics remain input features, just not prediction targets |
| 1-12 step horizons | 1-10 step horizons (15s-150s) | Pre-existing default in the implementation, not revisited during this fix pass |
| No feature engineering specified | Lag diffs (1,2,3) + rolling mean/std(3) added on top of the 7 raw metrics → 27 total input features | Reasonable ML practice, added independently of the proposal; flagged in-notebook (Step 3b) as an intentional deviation |
| Single multi-output GRU | 10 independent single-horizon GRUs | See architecture decision above |

### Storage/compute optimization (Phase 1 fix)
The sequence generator used to write a **full separate copy of the input window array
(X) for every one of the 10 horizons**, even though X only depends on the 240-step
lookback window, not on which horizon is being predicted — 9 of the 10 copies were
near-duplicates. Measured on a real Colab run before the fix (horizon 1 alone): train X
= 4,306 MB, val X = 1,324 MB, test X = 1,324 MB. Extrapolated across all 10 horizons:
**~69 GB total** for one dataset.

After the fix (X generated once per split; y stored as one shared `(N, 10, 4)` array
covering all 10 horizons instead of 10 separate y files): **~13 GB total** for the same
dataset — roughly a **5-7x** storage reduction, with a proportional reduction in Drive
I/O during training. Verified by executing the actual sequence-generation code
end-to-end on synthetic data: file count came out to 9 `.npy` files total (3 splits ×
X/y/containers) instead of 90 (3 splits × 10 horizons × 3 arrays).

### Phase 4 and Phase 5 — explicitly out of scope for this fix pass
Everything from "Phase 3: Adaptive Sliding Window" onward in this document remains
**unimplemented planning**, unchanged from the original proposal. This fix pass only
touched the notebook's Phase 1 (preprocessing/sequences) and Phase 3 (training loop —
see the phase-numbering note above). Don't assume any of the Phase 3/4/5 sections below
reflect implemented behavior — they're still exactly what they were when first written:
a plan, not a status report.

---

## Research Log: V1–V5 Root-Cause Investigation (post Phase 1-3 fix)

> Summarizes a full post-mortem review of `final_notebook/run/v1` through `v5`,
> each a real executed Colab run, not a code-only diff. Full detail lives in the
> conversation history behind this document; this is the durable summary.

**Version timeline** (each version fixed one genuinely evidenced defect, not a guess):

| Version | Result | Root cause fixed |
|---|---|---|
| V1 | No training ever ran | Dead duplicate Phase 3 code block; non-sequential kernel execution |
| V2 | 100% of data lost | Case identity assumed to be filename-encoded; real layout is folder-path-encoded (`complex/case1/container/kpi_*.csv`) |
| V3 | 75% of case data silently lost | `pivot_table(index=['timestamp','cmdb_id'])` collapsed all 4 cases' rows for any shared (timestamp, cmdb_id) pair, silently discarding 3 of 4 cases' actual values via `aggfunc='first'`. Predates all case-awareness work in this project — was silently active from the start. Fixed by adding `case_source` to the pivot index. |
| V4 | First complete, valid 10-horizon run | (no new defect — first run on the V3-corrected pipeline) |
| V5 | Reproduces V4 exactly (same seed/data) | Confirms reproducibility; contributes no new experimental signal |

**Headline finding (V4/V5):** 3 of 4 target variables (`mem_usage`, `mem_working_set`,
`mem_rss`) already meet the ≤10% MAPE research target (0.60–0.76% MAPE). The entire
"25–45% average MAPE" failure is driven almost exclusively by `cpu_usage`
(mean 133.5% MAPE) — a persistence (last-value) baseline beats the trained GRU on this
target on every horizon (0.07–0.43% MAPE vs. the GRU's 25–45%).

**Leading root-cause hypothesis (High confidence, not yet empirically confirmed):**
`cpu_usage` is trained as the raw cumulative Prometheus counter
(`container_cpu_usage_seconds_total`), whose absolute magnitude depends on how long a
container has been accumulating time before the observation window. Train's cases show
a ~30x larger scale (mean 21,168, std 24,814) than val's (mean 669, std 542) for this
one column; the 3 memory targets (gauges, not counters) show no comparable gap and pass
cleanly. Global z-score normalization, correct for gauges, miscalibrates this one
cumulative signal across cases.

**Open question (Medium confidence, unresolved):** test-set MAPE (254–522%) runs ~10x
worse than val MAPE (25–45%) on every horizon, despite test's raw CPU scale looking
*similar* to val's — meaning the scale-mismatch story above likely doesn't fully explain
test's performance on its own. Under investigation (Step 21 in the pipeline notebook).

**Status of the approved fix:** a controlled, single-variable experiment (predict CPU as
a delta — `future_cpu − last_observed_cpu` — instead of the raw counter, Horizon 1 only,
baseline checkpoint untouched) has been designed, code-reviewed, and integrated into the
notebook (Step 20). **It has not yet been executed against real data** — this is the
current blocking step before any claim about whether the fix works.

**WBS status** (full task list precedes this section in the project's working history):

| Task | Status |
|---|---|
| 1 — Fix #1 controlled experiment | Implemented (Step 20), **not yet run** |
| 2 — Case-heterogeneity investigation | Implemented (Step 21), **not yet run** |
| 3 — Per-target persistence baseline | Implemented (Step 22), **not yet run** |
| 4 — Multi-seed variance check | Implemented (Step 23), **not yet run** |
| 5 — Roll out fix to all 10 horizons | Gated on Task 1 passing — not started |
| 6 — Full val+test re-validation | Validation matrix extended with test-MAPE checks (Step 19); full re-run gated on Task 5 |
| 7 — Fallback: per-container normalization | Gated on Tasks 1/2 outcome — not started |
| 8 — Documentation | This section |
| 10 — Pre-flight integrity checks (guards against the V2/V3-style regressions recurring) | Implemented (Step 2b), verified to correctly detect both historical regression patterns |

---

## Proposal Requirements Mapping

### From Your Project Instructions
> "GRU-based forecasting with drift-aware mechanisms"

**5 Core Requirements:**
1. ✓ **Data preprocessing** (DONE - we're here)
2. ⏳ **GRU model architecture** (NEXT)
3. ⏳ **Adaptive sliding window** (AFTER GRU)
4. ⏳ **Drift detection** (AFTER WINDOW)
5. ⏳ **Incremental learning** (FINAL)

**3 Key Novelties:**
1. ⏳ **Drift-aware prediction** (Requires steps 2-5)
2. ⏳ **Burst-aware forecasting** (Requires GRU + threshold adaptation)
3. ⏳ **Multi-metric correlation** (Requires 7-metric GRU)

---

## Next 5 Phases (No Implementation Yet - Just Explanation)

### Phase 1: Sequence Generation
**Purpose:** Transform time-series data into ML-ready sequences

**What it does:**
- Takes 533k rows of continuous data
- Splits into 240-timestep windows (= 60 minutes of history)
- Creates prediction targets (1-12 steps ahead)
- Generates ~393,000 training sequences

**Input:**
```
training_data_with_container_norm.csv
(per-container normalized)
```

**Output:**
```
X_train: (N_sequences, 240 timesteps, 7 metrics)
y_train_1: (N_sequences, 1) - predict 1 step ahead (15 seconds)
y_train_2: (N_sequences, 1) - predict 2 steps ahead (30 seconds)
...
y_train_12: (N_sequences, 1) - predict 12 steps ahead (180 seconds)
```

**Why 240 timesteps?**
- 240 × 15 seconds = 3,600 seconds = 60 minutes
- Gives GRU 1-hour of history to make next prediction
- Balance between: too short (not enough context) vs. too long (too much data)

**Why 1-12 steps?**
- Proposal requires "short-term prediction" (next few minutes)
- 1 step = immediate next value (15 sec)
- 12 steps = far enough for useful forecasting (3 minutes)
- Covers: immediate actions (step 1) to planning window (step 12)

**Key decision:** How to split containers into train/val/test?
- **Option A:** Temporal split by case
  - Train: complex_case2 + single_case2
  - Val: single_case1
  - Test: complex_case1
  - ✓ Respects time order (better for drift detection)

- **Option B:** Random per-container split
  - 70% of each container → train
  - 15% → val
  - 15% → test
  - ✗ Mixes time periods (not good for drift)

**Recommendation:** Use Option A (temporal by case) - aligns with drift-aware goal

---

### Phase 2: GRU Model Architecture
**Purpose:** Build neural network for multi-step time-series forecasting

**What it does:**
- Learns temporal patterns from normalized sequences
- Predicts 1-12 steps ahead (multi-output)
- Processes 7 metrics simultaneously (multi-metric)
- Captures container-specific dynamics

**Model Structure (High-Level):**
```
Input Layer:
  Shape: (batch_size, 240 timesteps, 7 metrics)
  
GRU Layers:
  Layer 1: GRU(64 units, return_sequences=True)
    → Processes all 240 timesteps, passes to next layer
    → Learns temporal patterns across all 7 metrics
    → Output: (batch, 240, 64)
  
  Layer 2: GRU(32 units, return_sequences=False)
    → Processes sequence, outputs final hidden state
    → Compresses 240-timestep memory into 32-dim vector
    → Output: (batch, 32)

Output Layers (Multi-step forecasting):
  Dense(1) for 1-step ahead → predictions for step 1
  Dense(1) for 2-step ahead → predictions for step 2
  ...
  Dense(1) for 12-step ahead → predictions for step 12
  
  Total output: (batch, 12) - forecast 12 steps at once
```

**Why GRU (not LSTM)?**
- Proposal specifies GRU for efficiency
- Fewer parameters than LSTM (faster training/inference)
- Gating mechanism sufficient for this task
- Good balance: complexity vs. performance

**Why 240 timesteps as context?**
- Captures 1 hour of pattern
- Enough for: daily cycles, load patterns, anomalies
- Not too long: manageable compute, no distant irrelevant history

**Why 7 metrics together?**
- Proposal requires: CPU, memory, network, disk (7 total)
- Metrics are correlated (if CPU up, memory might up)
- Single GRU learns these correlations implicitly
- No separate models per metric

**Training approach:**
- Loss function: MSE (mean squared error)
- Optimizer: Adam (adaptive learning rate)
- Metrics: MAE, RMSE, MAPE (for evaluation)
- Early stopping: Stop if validation error increases

**Key question:** Separate models or single model?
- **Option A:** One GRU, output all 12 steps at once
  - ✓ Efficient
  - ✓ Captures multi-step dependencies
  - ✗ Harder to update individual steps

- **Option B:** Recursive (predict 1, use as input for 2, etc.)
  - ✓ More flexible
  - ✗ Error compounding (1 wrong → 2,3,4... wrong)

**Recommendation:** Option A (single output, all 12 steps) - cleaner, faster

---

### Phase 3: Adaptive Sliding Window
**Purpose:** Dynamically adjust prediction window based on workload variability

**From Proposal:**
> "Uses 500–1000 most recent samples. Window size dynamically adjusts based on detected workload variability (higher variability → larger window for stability)."

**What it does:**
- Monitors recent prediction errors
- If variability HIGH → increase window to 1000 samples
- If variability LOW → decrease window to 500 samples
- Goal: Capture enough history for high-variability containers, efficient for stable ones

**How it works:**
```
Step 1: Calculate recent error variability
  - Track last N prediction errors
  - Compute rolling std of errors
  - If std HIGH → workload is volatile
  - If std LOW → workload is stable

Step 2: Adjust window size
  - IF error_std > threshold_high:
      window_size = 1000  (need more history)
  - ELIF error_std < threshold_low:
      window_size = 500   (less history needed)
  - ELSE:
      keep current

Step 3: Apply to sequences
  - When predicting container_13 (variable): use 1000 steps
  - When predicting container_25 (stable): use 500 steps
  - Adjust window_size dynamically each prediction
```

**Why adaptive?**
- Proposal identifies burst-aware forecasting as key novelty
- Bursts = sudden variability increase
- More history needed during bursts
- Stable periods can use less history (efficient)

**Trade-offs:**
- Larger window: More compute, better context during bursts
- Smaller window: Faster, cleaner (no old irrelevant data)

**Implementation question:**
- Window adjustment per-container or global?
- **Per-container:** More precise, complex
- **Global:** Simpler, still effective

**Recommendation:** Per-container (since we have container-specific stats)

---

### Phase 4: Concept Drift Detection
**Purpose:** Detect when container behavior fundamentally changes

**From Proposal:**
> "Monitors prediction error using moving average and statistical checks. Triggers adaptation when drift is detected."

**What it does:**
- Tracks model performance over time
- Detects when patterns change (drift)
- Triggers retraining before accuracy collapses
- Maintains accuracy under evolving workloads

**Detection mechanism:**
```
Step 1: Track prediction errors
  - Every prediction, store error: error = |predicted - actual|
  - Maintain sliding window of recent errors (e.g., last 100)
  - Calculate: mean_error, std_error, trend

Step 2: Detect statistical change
  - Moving average: avg_error_recent vs. avg_error_baseline
  - If avg_error_recent > baseline + 2×std:
      → DRIFT DETECTED

Step 3: Trigger adaptation
  - If drift detected: increment adaptation counter
  - When counter reaches threshold: retrain model
  - Or: trigger incremental learning (see Phase 5)

Step 4: Case transition handling
  - Cases represent different system states
  - When moving case2→case1 (from data):
      → Reset error baseline
      → May see temporary error spike
      → Detect if it's real drift or just case change
```

**Drift types to detect:**
1. **Gradual drift:** Error slowly increases
   - Example: Container gradually handling more load
   - Solution: Fine-tune on recent data

2. **Sudden drift:** Error suddenly spikes
   - Example: Container app restart, config change
   - Solution: Full retrain with new pattern

3. **Cyclic behavior:** Error increases then decreases
   - Example: Daily pattern, weekly pattern
   - Solution: Don't retrain, just adjust thresholds

**Key metric: MAPE (Mean Absolute Percentage Error)**
- Relative error: (|predicted - actual| / |actual|) × 100%
- Better than MAE for values that vary widely
- Example: predicting 0.5 vs 50 requires different thresholds

**Proposal alignment:**
- Requires: moving average + statistical checks ✓
- Requires: triggers adaptation ✓
- Should detect: both gradual and sudden drift ✓

---

### Phase 5: Incremental Learning & Adaptive Thresholding
**Purpose:** Continuously adapt model to new patterns without full retraining

**From Proposal:**
> "Model updates continuously via error-triggered retraining or fine-tuning (small updates instead of full retraining)"

> "Dynamically adjusts prediction confidence intervals and alert thresholds based on recent error patterns"

**What it does:**

**Part A: Incremental Learning**
```
Traditional ML:
  - Train once on historical data
  - Deploy, never change
  - Accuracy degrades over time ✗

Online/Incremental Learning:
  - Train initially on historical data
  - In production: update continuously
  - Two strategies:

Strategy 1: Full retraining (expensive)
  - When drift detected
  - Retrain on new data + old data
  - Takes minutes, full recompute
  - Used for major shifts

Strategy 2: Fine-tuning (cheap)
  - Learning rate: very small (0.0001)
  - Gradient updates only
  - Takes seconds, updates weights slightly
  - Used for minor adjustments
  - Proposal preference: "small updates instead of full retraining"
```

**Example timeline:**
```
Day 1: Train on 533k rows (30 min)
Day 2: Collect 5k new rows, fine-tune (2 min)
Day 3: Collect 5k new rows, fine-tune (2 min)
Day 4: Collect 5k new rows, fine-tune (2 min)
...
Day 30: Detect major drift, full retrain (30 min) + 5 fine-tunes (10 min total)
```

**Part B: Adaptive Thresholding**
```
Static approach (traditional):
  - Fixed threshold: if error > 2.0 → alert
  - Problem: threshold irrelevant if patterns change
  
Adaptive approach (proposal):
  - Track recent error distribution
  - Threshold = recent_mean + 2×recent_std
  - Adjusts automatically as workload changes
  
Example:
  - Container_13 (variable) might have threshold = 0.8
  - Container_25 (stable) might have threshold = 0.15
  - Both adjust as behavior evolves
```

**Integration:**
```
Drift detection (Phase 4)
    ↓
Error above threshold?
    ↓
YES → Trigger incremental learning
      - Collect recent data
      - Fine-tune model (small learning rate)
      - Update adaptive thresholds
      - Reset error baseline
    ↓
Continue monitoring
```

**Timeline for all phases:**
- Every prediction: Calculate error, update moving average
- Every 100 predictions: Check drift, adjust window size
- If drift detected: Fine-tune model (incremental update)
- Per hour/day: Recalculate adaptive thresholds
- Per week/month: Full retraining check

**Key challenge:** Balance between:
- Too aggressive: Overfits to recent noise, unstable
- Too conservative: Doesn't adapt to real changes, accuracy drops

---

## Summary: Complete Pipeline

```
PHASE 1: Sequence Generation
Input:  training_data_with_container_norm.csv
Output: Training sequences (240-step windows, 1-12 step targets)
Goal:   Prepare data for neural network

PHASE 2: GRU Model
Input:  Training sequences
Output: Trained model + baseline metrics (MAE, RMSE, MAPE)
Goal:   Learn temporal patterns across all containers/metrics

PHASE 3: Adaptive Sliding Window
Input:  Prediction errors
Output: Dynamic window sizes per container
Goal:   Adjust context based on workload volatility

PHASE 4: Drift Detection
Input:  Prediction errors over time
Output: Drift signals, retraining triggers
Goal:   Detect when patterns change fundamentally

PHASE 5: Incremental Learning + Adaptive Thresholds
Input:  Drift signals, recent data
Output: Fine-tuned model, updated thresholds
Goal:   Continuously adapt without full retraining
```

---

## Key Design Decisions (Your Proposal Specifies These)

### 1. Prediction Horizon
- **What:** Forecast 1-12 steps ahead
- **In seconds:** 15-180 seconds (immediate to 3 minutes)
- **Why:** "Short-term prediction" for auto-scaling decisions
- **Decision:** Multi-output GRU (all 12 steps at once)

### 2. Context Window
- **Base:** 240 timesteps = 60 minutes
- **Adaptive:** 500-1000 samples based on variability
- **Why:** Balance historical context vs. computational efficiency
- **Decision:** Per-container adaptation

### 3. Metrics
- **Count:** 7 metrics (CPU: 3, Memory: 4)
- **Method:** Multi-metric GRU (shared hidden states)
- **Why:** Proposal requires capturing CPU-memory-network-disk correlations
- **Decision:** Single 7-input GRU (not separate models)

### 4. Drift Handling
- **Detection:** Error-based (moving average + statistical checks)
- **Response:** Incremental learning (fine-tuning, not full retrain)
- **Threshold:** Adaptive (per-container, per-metric)
- **Why:** Real systems change gradually; online learning needed
- **Decision:** Two-tier approach (drift detect → fine-tune)

### 5. Containers Handling
- **Approach:** Per-container normalization
- **Model sharing:** Single GRU trained on all containers
- **Why:** Containers behave similarly (same application, different load)
- **Decision:** Container ID as potential input feature (for future enhancement)

---

## What's Different From Generic Time-Series Forecasting?

**Generic approach:**
```
Historical data → Train model → Deploy → Predict
(Static, one-time)
```

**Your proposal (drift-aware):**
```
Historical data → Train model → Deploy ↓
                                        ├─→ Monitor errors
                                        ├─→ Detect drift
                                        ├─→ Adapt model
                                        └─→ Continuously improve
(Dynamic, online learning)
```

**Why this matters for containers:**
- Container workloads change daily
- Apps scale up/down
- Deployments happen
- Resource limits change
- Traditional static models fail quickly
- Need continuous adaptation

---

## Success Metrics (Evaluation)

**From Proposal:**
> "MAE / RMSE / MAPE for prediction accuracy. Drift adaptation speed (how quickly error returns to baseline after change). False positive/negative rate on proactive scaling decisions."

### 1. Prediction Accuracy (Phase 2)
- MAE: Average absolute error (what's mean offset?)
- RMSE: Root mean squared error (penalizes big errors)
- MAPE: Percentage error (relative accuracy)
- **Target:** ±10% MAPE on validation set

### 2. Adaptation Speed (Phases 4-5)
- Metric: Time to recover after drift
- Scenario: Workload suddenly changes
- Expected: Error spikes, then decreases within minutes
- **Target:** Recovery within 5-10 predictions (1-2 minutes)

### 3. Scaling Decisions (Production)
- False positives: Alert when no scaling needed
- False negatives: Miss when scaling is needed
- **Target:** <5% false positive rate, <10% false negative

### 4. Resource Efficiency
- Model size: Should fit in memory
- Inference time: Should be <100ms per prediction
- Retraining time: Fine-tune should be <1 minute

---

## What You're Building

**Not:**
- ✗ Generic LSTM for time-series
- ✗ Static forecasting model
- ✗ Single-container predictor

**But:**
- ✓ Multi-container adaptive forecasting system
- ✓ Drift-aware with automatic adaptation
- ✓ Real-time friendly (fast inference + incremental updates)
- ✓ Production-ready for Kubernetes/Docker auto-scaling

---

## Timeline Estimate (No Implementation Yet)

- Phase 1 (Sequences): 1-2 hours coding
- Phase 2 (GRU): 2-3 hours (including hyperparameter tuning)
- Phase 3 (Window): 1-2 hours
- Phase 4 (Drift): 2-3 hours (statistical testing important)
- Phase 5 (Learning): 3-4 hours (tricky to get right)

**Total:** ~10-15 hours of implementation
**Timeline:** 1-2 weeks if done incrementally with testing

---

## Next Question for You

Based on this roadmap, should we proceed in order (Phases 1-5), or do you want to:
- A) Implement Phase 1 first (sequences) - foundation for rest
- B) Skip straight to Phase 2 (GRU) - make sure model works
- C) Start with Phase 4 (drift detection) - understand detection first
- D) Something else based on your priorities?
