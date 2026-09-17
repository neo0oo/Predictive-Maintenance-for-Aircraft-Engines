import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import load_model

ARTIFACT_DIR = Path("models/artifacts/FD001")

prep = joblib.load(ARTIFACT_DIR / "preprocessing.joblib")
tree_model = joblib.load(ARTIFACT_DIR / "tree_model.joblib")
lstm_model = load_model(ARTIFACT_DIR / "lstm_model.keras")

test_df = pd.read_csv("data/processed/test_FD001_clean.csv")
rul_df = pd.read_csv("data/processed/RUL_FD001_clean.csv")

sensor_cols = prep['sensor_cols']
roll_window = prep['roll_window']

test_df = test_df.sort_values(['unit', 'cycle']).reset_index(drop=True)
grouped = test_df.groupby('unit')[sensor_cols]
roll_mean = grouped.rolling(window=roll_window, min_periods=1).mean().reset_index(level=0, drop=True)
roll_std = grouped.rolling(window=roll_window, min_periods=1).std().fillna(0).reset_index(level=0, drop=True)
roll_mean.columns = [f'{c}_rollmean' for c in sensor_cols]
roll_std.columns = [f'{c}_rollstd' for c in sensor_cols]
test_df = pd.concat([test_df, roll_mean, roll_std], axis=1)

units = sorted(test_df['unit'].unique())
assert units == list(range(1, len(units) + 1)), "expected contiguous unit ids 1..N matching RUL file row order"
# the test set is truncated before failure, so RUL_FD001.txt is capped the same way
# the training target was - scoring against an uncapped true value would unfairly
# penalize the model for a target it was deliberately trained not to chase
true_rul = np.clip(rul_df['RUL'].values, a_min=None, a_max=prep['cap'])

last_rows = test_df.sort_values('cycle').groupby('unit').tail(1).sort_values('unit')
X_test_tree = last_rows[prep['tree_feature_cols']]
tree_preds = tree_model.predict(X_test_tree)

tree_rmse = np.sqrt(mean_squared_error(true_rul, tree_preds))
tree_r2 = r2_score(true_rul, tree_preds)
print(f"Random Forest Test RMSE: {tree_rmse:.2f} cycles")
print(f"Random Forest Test R²: {tree_r2:.4f}")


def make_last_window(data, window, cols):
    X = []
    for unit_id in sorted(data['unit'].unique()):
        g = data[data['unit'] == unit_id].sort_values('cycle')
        values = g[cols].values
        if len(g) < window:
            pad_len = window - len(g)
            pad = np.repeat(values[0:1], pad_len, axis=0)
            values = np.vstack([pad, values])
        X.append(values[-window:])
    return np.array(X)


lstm_feature_cols = prep['lstm_feature_cols']
test_scaled = test_df.copy()
test_scaled[lstm_feature_cols] = prep['lstm_scaler'].transform(test_scaled[lstm_feature_cols])

X_test_lstm = make_last_window(test_scaled, prep['window'], lstm_feature_cols)
lstm_preds = lstm_model.predict(X_test_lstm).flatten()

lstm_rmse = np.sqrt(mean_squared_error(true_rul, lstm_preds))
lstm_r2 = r2_score(true_rul, lstm_preds)
print(f"LSTM Test RMSE: {lstm_rmse:.2f} cycles")
print(f"LSTM Test R²: {lstm_r2:.4f}")
