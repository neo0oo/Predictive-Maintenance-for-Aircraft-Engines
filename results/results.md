# Model Results

RUL is capped at 125 cycles for all approaches (standard for this dataset — early-life
RUL is roughly constant/unknowable, so capping keeps the model from being penalized for
not predicting a precise number thousands of cycles before failure).

Lower RMSE is better; RMSE is in cycles. Higher R² is better.

## FD001 — 1 operating condition, 1 fault mode

| Approach | RMSE | R² |
|---|---|---|
| RF (single split) | 15.25 | 0.8664 |
| RF (5-fold GroupKFold CV, avg) | 16.71 ± 0.63 | 0.8389 ± 0.01 |
| LSTM (30-cycle window) | 15.05 | 0.8702 |

<details>
<summary>GroupKFold per-fold breakdown</summary>

| Fold | RMSE | R² |
|---|---|---|
| 1 | 16.74 | 0.8386 |
| 2 | 16.59 | 0.8415 |
| 3 | 16.70 | 0.8395 |
| 4 | 15.78 | 0.8567 |
| 5 | 17.76 | 0.8183 |

</details>

Notes: LSTM trained up to 100 epochs with early stopping (patience=10 on val_loss);
this run stopped at epoch 38. The LSTM isn't seeded, so its exact numbers will drift
slightly (~±0.5 RMSE) between runs — the RF numbers are seeded (`random_state=42`) and
reproducible.

## FD002 — 6 operating conditions, 1 fault mode

*Not yet trained.*

## FD003 — 1 operating condition, 2 fault modes

*Not yet trained.*

## FD004 — 6 operating conditions, 2 fault modes

*Not yet trained.*
