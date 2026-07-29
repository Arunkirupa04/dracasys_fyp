# Future Work

1. **Huber or weighted MSE** combined with R3 scaling — test interaction between robust loss and robust input scale.

2. **Learning rate sweep under R3** — if conditioning improves, a lower LR might reduce plateau behaviour seen in production.

3. **Per-layer gradient norm logging** — directly measure optimisation differences between R0 and R3 during training.

4. **Extended horizons** — Day-2 residual propagation under different scalings.

5. **Production cohort** — replicate RRE on the 435-container production split if research findings warrant it.

6. **Architecture search** — only after input scaling hypothesis is resolved; avoid confounding variables.

7. **Theoretical analysis** — condition number of sequence covariance under R0 vs R3 scaling.

If RRE v1.0 is null, future effort should shift away from residual preprocessing toward Prophet improvements or alternative sequence models.
