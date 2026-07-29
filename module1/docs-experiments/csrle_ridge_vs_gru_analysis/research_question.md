# Research Question

## Primary

> **Why did Ridge outperform the frozen Hybrid GRU on the B0 positive-control despite B0 being intentionally designed as a nonlinear temporal process?**

## Context (frozen CSRLE Stage 2)

On B0-LC (deterministic positive control), cohort-mean **within-trajectory residual Pearson r** was approximately:

- **Ridge:** ~0.158  
- **GRU:** ~0.084  
- **Condition A (real data):** ~0.016  

MAE was similar across GRU, Ridge, and zero predictor (~0.094). The gap is primarily **correlation and variance recovery**, not raw error magnitude.

## What this analysis is not

- Not a model-improvement study  
- Not a retuning of GRU or CSRLE  
- Not new synthetic data generation  

## Success criteria

Provide **quantitative, evidence-backed** explanations covering:

1. Linear vs nonlinear structure in B0 Prophet residuals  
2. GRU variance collapse vs Ridge amplitude recovery  
3. Frequency-domain preservation  
4. MSE training objective effects  
5. Link to weak real-data (Condition A) residual learning  
