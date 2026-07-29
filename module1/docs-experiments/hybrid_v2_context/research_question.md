# Research Question

## Primary

**Which contextual information actually improves Hybrid residual learning beyond the baseline residual-only input?**

## Sub-questions

1. Does Prophet forecast level (`prophet_yhat_scaled`) improve residual learning?
2. Does explicit local volatility (`res_roll_std_12`) help beyond implicit GRU memory?
3. Does container scale prior (`cpu_std`) calibrate corrections?
4. Does peak regime flag (`is_peak_p90`) improve peak-period residual prediction?
5. What is the **minimum** context set that matches full V4 performance?

## Success criteria

Improvement is measured on **residual Pearson r**, **std ratio**, and **Day-1 CPU MAE** with paired bootstrap CIs and Wilcoxon tests — not feature count.

## Interpretation rule

If V1 or V2 matches V4 within noise → recommend simpler variant (parsimony).
