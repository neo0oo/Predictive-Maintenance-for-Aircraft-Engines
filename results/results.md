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

| Approach | RMSE | R² |
|---|---|---|
| XGBoost (single split, tuned) | 18.54 | 0.8031 |
| XGBoost (5-fold CV, avg) | 18.58 ± 0.71 | 0.8008 ± 0.02 |
| LSTM (window search) | 13.95 | 0.8852 |

<details>
<summary>CV per-fold breakdown</summary>

| Fold | RMSE | R² |
|---|---|---|
| 1 | 18.54 | 0.8031 |
| 2 | 19.95 | 0.7709 |
| 3 | 18.13 | 0.8102 |
| 4 | 17.95 | 0.8135 |
| 5 | 18.31 | 0.8066 |

</details>

## FD003 — 1 operating condition, 2 fault modes

| Approach | RMSE | R² |
|---|---|---|
| XGBoost (single split, tuned) | 15.83 | 0.8509 |
| XGBoost (5-fold CV, avg) | 16.56 ± 1.19 | 0.8318 ± 0.03 |
| LSTM (window search) | 14.80 | 0.8738 |

<details>
<summary>CV per-fold breakdown</summary>

| Fold | RMSE | R² |
|---|---|---|
| 1 | 15.83 | 0.8509 |
| 2 | 16.08 | 0.8364 |
| 3 | 18.85 | 0.7727 |
| 4 | 16.50 | 0.8410 |
| 5 | 15.53 | 0.8581 |

</details>


## FD004 — 6 operating conditions, 2 fault modes

| Approach | RMSE | R² |
|---|---|---|
| XGBoost (single split, tuned) | 19.17 | 0.7801 |
| XGBoost (5-fold CV, avg) | 17.02 ± 1.41 | 0.8238 ± 0.03 |
| LSTM (window search) | 19.50 | 0.7841 |

<details>
<summary>CV per-fold breakdown</summary>

| Fold | RMSE | R² |
|---|---|---|
| 1 | 19.17 | 0.7801 |
| 2 | 17.36 | 0.8181 |
| 3 | 17.18 | 0.8200 |
| 4 | 16.58 | 0.8354 |
| 5 | 14.80 | 0.8656 |

</details>

## Test Set Evaluation

Everything above is on an internal 80/20 validation split carved out of the training
engines. This section is the real held-out benchmark: the official `test_FD00x`
trajectories (which the models never saw at all) scored against `RUL_FD00x`, the true
answer key. Test trajectories are truncated mid-life rather than run to failure, so
each engine contributes exactly one prediction - from its last available cycle for the
tree model, from its last window of cycles for the LSTM - compared against that
engine's single true RUL value (also capped at 125, matching the training target).

### FD001

| Approach | RMSE | R² |
|---|---|---|
| Random Forest | 18.12 | 0.7955 |
| LSTM | 16.49 | 0.8306 |

### FD002

| Approach | RMSE | R² |
|---|---|---|
| XGBoost | 16.57 | 0.8510 |
| LSTM | 14.44 | 0.8870 |

### FD003

| Approach | RMSE | R² |
|---|---|---|
| XGBoost | 20.06 | 0.7377 |
| LSTM | 17.26 | 0.8057 |

### FD004

| Approach | RMSE | R² |
|---|---|---|
| XGBoost | 19.16 | 0.8013 |
| LSTM | 19.94 | 0.7847 |
