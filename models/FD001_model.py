import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv("data/processed/train_FD001_clean.csv")

# rolling-window sensor features (mean/std over the last 5 cycles per engine),
# computed before any split so both training flows below can use them
window = 5
sensor_cols = [c for c in df.columns if c.startswith('sensor_')]

df = df.sort_values(['unit', 'cycle']).reset_index(drop=True)
grouped = df.groupby('unit')[sensor_cols]

roll_mean = grouped.rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
roll_std = grouped.rolling(window=window, min_periods=1).std().fillna(0).reset_index(level=0, drop=True)

roll_mean.columns = [f'{c}_rollmean' for c in sensor_cols]
roll_std.columns = [f'{c}_rollstd' for c in sensor_cols]

df = pd.concat([df, roll_mean, roll_std], axis=1)

#we have to split the data into training and validating sets, doing this by solely splitting the engine ID column

unit_IDs = df['unit'].unique()
train_units, val_units = train_test_split(unit_IDs, test_size=0.2, random_state=42)

train_df = df[df['unit'].isin(train_units)].reset_index(drop=True)
val_df = df[df['unit'].isin(val_units)].reset_index(drop=True)

# capping the RUL values to 125 to remove noise
CAP = 125
train_df['RUL'] = train_df['RUL'].clip(upper=CAP)
val_df['RUL'] = val_df['RUL'].clip(upper=CAP)

feature_cols = [c for c in train_df.columns if c not in ['unit', 'RUL']]
tree_feature_cols = list(feature_cols)

X_train = train_df[feature_cols]
y_train = train_df['RUL']

X_val = val_df[feature_cols]
y_val = val_df['RUL']

# training the baseline random forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# making predictions on the validation set
preds = rf_model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, preds))
print(f"Validation RMSE: {rmse:.2f} cycles")
r_2 = r2_score(y_val, preds)
print(f"Validation R^2: {r_2:.4f}")

# 5-fold GroupKFold cross-validation instead of a single split

df['RUL'] = df['RUL'].clip(upper=CAP)

feature_cols = [c for c in df.columns if c not in ['unit', 'RUL']]
X = df[feature_cols]
y = df['RUL']
groups = df['unit']

gkf = GroupKFold(n_splits=5)
rmse_scores, r2_scores = [], []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2 = r2_score(y_val, preds)
    rmse_scores.append(rmse)
    r2_scores.append(r2)
    print(f"Fold {fold+1}: RMSE={rmse:.2f}, R²={r2:.4f}")

print(f"\nAverage RMSE: {np.mean(rmse_scores):.2f} (+/- {np.std(rmse_scores):.2f})")
print(f"Average R²: {np.mean(r2_scores):.4f} (+/- {np.std(r2_scores):.2f})")

# deep learning model using LSTM

WINDOW = 30 
feature_cols = [c for c in df.columns if c not in ['unit', 'cycle', 'RUL']]
scaler = StandardScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
val_df[feature_cols] = scaler.transform(val_df[feature_cols])

def make_windows(data, window=WINDOW):
    X, y = [], []
    for unit_id, g in data.groupby('unit'):
        g = g.sort_values('cycle')
        values = g[feature_cols].values
        ruls = g['RUL'].values

        if len(g) < window:
            # pad short engines by repeating the first row
            pad_len = window - len(g)
            pad = np.repeat(values[0:1], pad_len, axis=0)
            values = np.vstack([pad, values])
            ruls = np.concatenate([np.full(pad_len, ruls[0]), ruls])

        for i in range(len(values) - window + 1):
            X.append(values[i:i+window])
            y.append(ruls[i+window-1])

    return np.array(X), np.array(y)

X_train, y_train = make_windows(train_df)
X_val, y_val = make_windows(val_df)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(WINDOW, X_train.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

preds = model.predict(X_val).flatten()
rmse = np.sqrt(mean_squared_error(y_val, preds))
r2 = r2_score(y_val, preds)
print(f"LSTM Validation RMSE: {rmse:.2f} cycles")
print(f"LSTM Validation R²: {r2:.4f}")

# --- persist trained models + preprocessing for later test-set evaluation ---
import joblib
from pathlib import Path

ARTIFACT_DIR = Path("models/artifacts/FD001")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(rf_model, ARTIFACT_DIR / "tree_model.joblib")
model.save(ARTIFACT_DIR / "lstm_model.keras")

joblib.dump({
    'sensor_cols': sensor_cols,
    'tree_feature_cols': tree_feature_cols,
    'lstm_feature_cols': feature_cols,
    'lstm_scaler': scaler,
    'window': WINDOW,
    'roll_window': window,
    'cap': CAP,
}, ARTIFACT_DIR / "preprocessing.joblib")

print(f"\nSaved trained models and preprocessing to {ARTIFACT_DIR}/")