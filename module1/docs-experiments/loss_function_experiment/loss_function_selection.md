# Loss Function Selection

## Decision summary

| Question | Answer |
|----------|--------|
| **Which loss?** | **Dispersion-Augmented MSE (DA-MSE)** |
| **Replace or augment MSE?** | **Augment** — keep MSE as the point-error term; add one dispersion penalty |
| **Why single choice?** | Minimal intervention that directly targets observed **under-dispersion** without architecture change or tautological correlation optimization |

---

## Recommended loss: DA-MSE

### Mathematical formulation

For one training example with target vector **y** ∈ ℝ^H and prediction **ŷ** ∈ ℝ^H (H = 96):

**1. Mean squared error (unchanged from baseline)**

\[
\mathcal{L}_{\text{MSE}} = \frac{1}{H} \sum_{h=1}^{H} (\hat{y}_h - y_h)^2
\]

**2. One-sided log-variance dispersion penalty**

Using sample standard deviations over the H-step trajectory (ddof = 0, **frozen**):

\[
\sigma_y = \text{std}(\mathbf{y}), \quad \sigma_{\hat{y}} = \text{std}(\hat{\mathbf{y}})
\]

\[
\delta = \log(\sigma_y + \epsilon) - \log(\sigma_{\hat{y}} + \epsilon)
\]

\[
\mathcal{L}_{\text{disp}} = \max(0, \delta)^2
\]

Penalizes **under-dispersion only** (σ̂ < σ_y). Does not penalize over-dispersion — matching the observed failure mode (collapse, not inflation).

**3. Combined objective (frozen λ = 1.0)**

\[
\mathcal{L}_{\text{DA-MSE}} = \mathcal{L}_{\text{MSE}} + \lambda \cdot \mathcal{L}_{\text{disp}}, \quad \lambda = 1.0
\]

**4. Numerical stability**

ε = 1e−6 (frozen) applied inside logarithms.

**5. Batch training**

Loss averaged over batch: \(\mathcal{L} = \frac{1}{B}\sum_{b=1}^{B} \mathcal{L}^{(b)}_{\text{DA-MSE}}\).

Mini-batch gradients flow through **per-sequence** σ̂ computed from each sequence’s 96 predicted steps (not batch-pooled σ̂ across mixed sequences — avoids batch-statistics confound).

---

## Why DA-MSE is the best choice for this hypothesis

### 1. Direct alignment with observed failure

CSRLE and Ridge analysis document **under-dispersion** (std ratio ≈ 0.07 vs Ridge ≈ 0.22), not generic poor MAE. DA-MSE adds the **exact missing incentive** MSE lacks: match temporal amplitude.

### 2. Minimal causal intervention

Among valid alternatives, DA-MSE changes **one interpretable degree of freedom** — dispersion penalty — while preserving MSE’s point-forecast role. This is the cleanest test of “is MSE the cause?” vs “should we abandon point-error training entirely?”

### 3. Avoids tautology

Optimizing **Pearson correlation directly** (1 − r) would confound the experiment: correlation is a **primary outcome metric**. Success would partially reflect optimizing what is measured. DA-MSE targets dispersion; correlation is **downstream evidence** of improved shape.

### 4. No architecture change

Unlike Gaussian NLL, β-NLL, or faithful heteroscedastic regression, DA-MSE requires **no variance output head** — keeping the GRU identically sized and preserving isolation of the loss hypothesis.

### 5. Literature consistency

Recent work on MSE+correlation plateaus identifies **dispersion–gradient coupling** as core to shape-learning failure. A log-variance penalty is a standard, scale-aware dispersion match (related to heteroscedastic calibration and moment-matching literature) without full probabilistic modeling.

### 6. Ridge-compatible interpretation

Ridge achieves higher std ratio with similar MAE. DA-MSE explicitly permits the model to **increase output amplitude** without abandoning squared-error fidelity — mirroring what Ridge accomplishes implicitly via linear lag fitting.

---

## Why MSE is augmented, not replaced

| Replace MSE with… | Problem |
|-------------------|---------|
| Pure correlation (1 − r) | Scale-invariant; does not require variance recovery; tautological for r |
| Pure MAE (L1) | Addresses robustness, not dispersion; confounds mechanism |
| Pure dispersion penalty | Ignores point accuracy; may over-inflate variance with poor MAE |
| Huber | Outlier robustness ≠ variance collapse |

**Augmentation preserves the baseline’s point-error optimum** while testing whether **adding the dispersion term MSE omits** resolves collapse. If only replacement worked, we could not distinguish “MSE is wrong” from “MSE is incomplete.”

---

## Rejected alternatives (with reasons)

| Candidate | Rejection rationale |
|-----------|---------------------|
| **Pearson / cosine loss (1 − r)** | Primary outcome metric; scale-invariant so std ratio may not improve; tautological |
| **MSE + λ(1 − r) joint loss** | Requires balancing λ; MSE gradient dominance documented; **two competing terms** → hyperparameter study risk; λ not uniquely identifiable a priori |
| **Gaussian NLL / heteroscedastic NLL** | Requires second output for σ̂(x); **architecture change** confounds hypothesis |
| **β-NLL (Seitzer et al., 2022)** | Same dual-output issue; designed for aleatoric uncertainty, not deterministic residual correction |
| **Quantile / pinball loss** | Estimates conditional quantiles, not mean; changes target of optimization |
| **Huber / smooth L1** | Robustness to outliers; no theoretical link to under-dispersion on zero-mean residuals |
| **MAE (L1) replacement** | Different inductive bias; not the mechanism Ridge analysis implicates |
| **Timestep-weighted MSE (peak-aware)** | Already evaluated project-wide with limited benefit; targets peaks not dispersion |
| **Batch inverse-variance weighting** | Requires per-label noise estimates not available for residuals |
| **Wasserstein / MMD losses** | Distribution matching overkill; harder to interpret; multi-hyperparameter |

**No compelling scientific reason** to run multiple alternative losses in one experiment — each adds multiplicity and dilutes causal interpretation. DA-MSE is the **single best targeted probe**.

---

## Frozen hyperparameters (loss-specific)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| λ | **1.0** | See **Explicit λ justification** below |
| ε | 1e−6 | Numerical stability |
| std ddof | 0 | Population std over 96 steps; frozen |
| Penalty direction | One-sided (under-dispersion only) | Matches observed collapse direction |
| λ sensitivity analysis | **Out of scope** | Deferred to [future_scope.md](future_scope.md) |

---

## Explicit λ justification (λ = 1.0)

### Why λ is frozen before implementation

λ is set to **1.0 in the protocol document**, prior to any training run. It is **not** selected by validation performance, grid search, or post-hoc tuning. This is intentional.

### Hypothesis test, not hyperparameter optimization

LFHE asks a **causal question**: does introducing dispersion-aware supervision (holding architecture, data, and evaluation fixed) change variance collapse and correlation? That requires **one** treatment level for the dispersion term. Treating λ as a tunable knob converts the study into a **multi-factor optimization problem** (loss family × λ), which:

- Weakens causal attribution (“best λ was found” vs “dispersion supervision was tested”),
- Invites multiple-comparison and overfitting to B0,
- Conflicts with the frozen CSRLE philosophy established in prior stages.

### Purpose of λ = 1.0 specifically

With \(\mathcal{L} = \mathcal{L}_{\text{MSE}} + \lambda \mathcal{L}_{\text{disp}}\) and \(\mathcal{L}_{\text{disp}} = \max(0, \log\sigma_y - \log\sigma_{\hat{y}})^2\):

- **λ = 1.0** assigns **unit nominal weight** to the log-variance mismatch when under-dispersed — the dispersion term enters at the **same scale as a single O(1) MSE contribution** on standardized residuals.
- This is a **principled default** for a first prospective test: neither zero (which would revert to pure MSE) nor an arbitrary large weight that would dominate point-error training.
- The experiment tests whether **any meaningful dispersion supervision** at equal nominal priority alters outcomes — sufficient to falsify “MSE alone is the bottleneck” vs “objective family is irrelevant.”

### Why optimal λ search is rejected

| Concern | Consequence of λ search |
|---------|-------------------------|
| Extra experimental variable | Cannot isolate dispersion **concept** from λ **magnitude** |
| Data snooping on B0 | Inflated false support for H₁ |
| Post-hoc rationalization | “We tuned λ” undermines pre-registration |
| Scope creep | Becomes accuracy optimization, not hypothesis test |

If λ = 1.0 yields null results, the correct inference is **H₀ not rejected at this supervision strength** — not “try another λ until it works.”

### Sensitivity analysis deferred

λ sweeps (e.g. 0.1, 0.5, 2.0, 5.0) belong in **Future Work** under a **new protocol amendment** (LFHE v2), not in v1. See [future_scope.md](future_scope.md).

---

## Expected gradient effect (qualitative)

When σ̂ ≪ σ_y, \(\mathcal{L}_{\text{disp}} > 0\) pushes predictions toward **higher cross-step variance** without requiring architectural capacity for heteroscedastic noise modeling. MSE gradient alone often favors **shrunk corrections** on small-magnitude standardized residuals.

---

## Answers to design checklist (loss-specific)

| # | Question | Answer |
|---|----------|--------|
| 1 | Which loss? | DA-MSE |
| 2 | Why best? | Minimal, non-tautological, dispersion-targeted, no architecture change |
| 3 | Why others rejected? | See table above |
| 4 | Replace or augment? | **Augment** MSE |
| 5 | Mathematical form? | \(\mathcal{L} = \mathcal{L}_{\text{MSE}} + \lambda \max(0, \log\sigma_y - \log\sigma_{\hat{y}})^2\) |
| 6 | Architecture identical? | **Yes** |
| 7 | Prophet identical? | **Yes** |
| 8 | CSRLE generators identical? | **Yes** |
| 9 | Datasets identical? | **Yes** |
| 10 | Evaluation identical? | **Yes** (CSRLE Stage 2 metric suite) |
