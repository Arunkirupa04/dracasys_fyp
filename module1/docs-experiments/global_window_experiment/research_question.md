# Research Questions

## Primary

> Does increasing the input history of the Global GRU improve long-term CPU forecasting because additional temporal memory exists beyond the current 96-step window?

## Secondary

| # | Question |
|---|----------|
| 1 | Is 96 steps actually insufficient for the Global model on raw CPU? |
| 2 | Does longer context improve **every** workload equally? |
| 3 | Do containers with **stronger long-range temporal dependence** (TMA metrics) benefit more? |
| 4 | Is there a **point of diminishing returns** (192 vs 288 vs 384)? |
| 5 | Does increasing the window increase **overfitting** risk given shorter effective sequence counts? |

## Non-goals

- Proving Hybrid should use longer windows (TMA already addressed — **no**)
- Replacing Prophet decomposition
- Joint Hybrid vs Global model selection (separate thesis narrative)

## Expected thesis contribution

Empirical test of TMA's **Global GRU recommendation**: long memory exists in CPU; does exposing it to the GRU improve forecasts?
