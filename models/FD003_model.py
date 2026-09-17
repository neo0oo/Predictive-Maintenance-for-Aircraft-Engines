import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold, KFold, RandomizedSearchCV
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

df = pd.read_csv("data/processed/train_FD003_clean.csv")

CAP = 125
df['RUL'] = df['RUL'].clip(upper=CAP)

SENSOR_COLS = [c for c in df.columns if c.startswith('sensor_')]

FLAT_CV_THRESHOLD = 1e-4
ROLL_WINDOW = 5


def compute_engine_trends(data, sensor_cols):
    """One row per engine: the linear slope of each sensor over that engine's life.
    This is the 'degradation signature' used to split engines by likely fault mode -
    HPC degradation and fan degradation don't push the same sensors in the same
    direction, so an engine's vector of per-sensor trends should separate the two
    fault modes better than any single cycle's raw readings would."""
    rows = []
    for unit_id, g in data.groupby('unit'):
        g = g.sort_values('cycle')
        slopes = {c: np.polyfit(g['cycle'], g[c], 1)[0] for c in sensor_cols}
        slopes['unit'] = unit_id
        rows.append(slopes)
    return pd.DataFrame(rows).set_index('unit')


def engineer_features(fit_df, *apply_dfs):
    """
    Fits informative-sensor selection and the fault-mode clustering on `fit_df`
    only, then applies them to fit_df and every frame in apply_dfs - same
    train-only-fit discipline used for FD002's regime clustering, and for the same
    reason: this gets refit inside every CV fold so a fold's held-out engines never
    influence the statistics used to describe them.
    """
    frames = [fit_df.copy()] + [f.copy() for f in apply_dfs]
    fit = frames[0]

    
    sensor_std = fit[SENSOR_COLS].std()
    sensor_mean = fit[SENSOR_COLS].mean().abs()
    cv = (sensor_std / sensor_mean).abs()
    near_constant = cv[cv < FLAT_CV_THRESHOLD].index.tolist()
    informative_sensors = [c for c in SENSOR_COLS if c not in near_constant]

    
    fit_trends = compute_engine_trends(fit, informative_sensors)
    trend_scaler = StandardScaler()
    fit_trends_scaled = trend_scaler.fit_transform(fit_trends)

    fault_kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    fit_labels = fault_kmeans.fit_predict(fit_trends_scaled)
    fit_cluster_map = pd.Series(fit_labels, index=fit_trends.index)

    cluster_maps = [fit_cluster_map]
    for f in apply_dfs:
        trends = compute_engine_trends(f, informative_sensors)
        labels = fault_kmeans.predict(trend_scaler.transform(trends))
        cluster_maps.append(pd.Series(labels, index=trends.index))

    for f, cmap in zip(frames, cluster_maps):
        f['fault_cluster'] = f['unit'].map(cmap)
        f['fault_cluster_0'] = (f['fault_cluster'] == 0).astype(int)
        f['fault_cluster_1'] = (f['fault_cluster'] == 1).astype(int)

    # rolling-window features (mean/std over the last ROLL_WINDOW cycles)
    def add_rolling(f):
        f = f.sort_values(['unit', 'cycle']).reset_index(drop=True)
        grouped = f.groupby('unit')[informative_sensors]
        roll_mean = grouped.rolling(window=ROLL_WINDOW, min_periods=1).mean().reset_index(level=0, drop=True)
        roll_std = grouped.rolling(window=ROLL_WINDOW, min_periods=1).std().fillna(0).reset_index(level=0, drop=True)
        roll_mean.columns = [f'{c}_rollmean' for c in informative_sensors]
        roll_std.columns = [f'{c}_rollstd' for c in informative_sensors]
        return pd.concat([f, roll_mean, roll_std], axis=1)

    frames = [add_rolling(f) for f in frames]

    feature_cols = (
        ['fault_cluster_0', 'fault_cluster_1']
        + informative_sensors
        + [f'{c}_rollmean' for c in informative_sensors]
        + [f'{c}_rollstd' for c in informative_sensors]
    )

    fitted = {'trend_scaler': trend_scaler, 'fault_kmeans': fault_kmeans}
    return frames, feature_cols, informative_sensors, near_constant, fit_cluster_map, fitted


# split by engine unit, same as FD001/FD002
unit_IDs = df['unit'].unique()
train_units, val_units = train_test_split(unit_IDs, test_size=0.2, random_state=42)

train_raw = df[df['unit'].isin(train_units)].reset_index(drop=True)
val_raw = df[df['unit'].isin(val_units)].reset_index(drop=True)

(train_df, val_df), feature_cols, informative_sensors, near_constant, train_cluster_map, fitted = engineer_features(train_raw, val_raw)

print("dropping sensors that are flat (relative CV):", near_constant)
print(f"informative sensors kept: {len(informative_sensors)}")
print("fault-mode cluster sizes (train engines):", train_cluster_map.value_counts().sort_index().to_dict())

X_train = train_df[feature_cols]
y_train = train_df['RUL']

X_val = val_df[feature_cols]
y_val = val_df['RUL']


param_dist = {
    'n_estimators': [200, 400, 600, 800],
    'max_depth': [3, 4, 5, 6, 8],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'reg_alpha': [0, 0.1, 1.0],
    'reg_lambda': [1.0, 2.0, 5.0],
}

search = RandomizedSearchCV(
    XGBRegressor(random_state=42, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=30,
    cv=GroupKFold(n_splits=5),
    scoring='neg_root_mean_squared_error',
    random_state=42,
    n_jobs=-1,
)
search.fit(X_train, y_train, groups=train_df['unit'])

print("best XGBoost params:", search.best_params_)

model = search.best_estimator_
preds = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, preds))
r2 = r2_score(y_val, preds)
print(f"XGBoost Validation RMSE: {rmse:.2f} cycles")
print(f"XGBoost Validation R²: {r2:.4f}")


cluster_preds = np.empty(len(val_df))
for cluster in [0, 1]:
    tr_mask = train_df['fault_cluster'] == cluster
    va_mask = val_df['fault_cluster'] == cluster

    cluster_model = XGBRegressor(**search.best_params_, random_state=42, n_jobs=-1)
    cluster_model.fit(X_train[tr_mask], y_train[tr_mask])
    cluster_preds[va_mask.values] = cluster_model.predict(X_val[va_mask])

cluster_rmse = np.sqrt(mean_squared_error(y_val, cluster_preds))
cluster_r2 = r2_score(y_val, cluster_preds)
print(f"Separate per-cluster models Validation RMSE: {cluster_rmse:.2f} cycles")
print(f"Separate per-cluster models Validation R²: {cluster_r2:.4f}")
print(f"(unified model with fault_cluster feature: RMSE={rmse:.2f}, R²={r2:.4f})")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores, r2_scores = [], []

for fold, (tr_idx, va_idx) in enumerate(kf.split(unit_IDs)):
    fold_train_units = unit_IDs[tr_idx]
    fold_val_units = unit_IDs[va_idx]

    fold_train_raw = df[df['unit'].isin(fold_train_units)].reset_index(drop=True)
    fold_val_raw = df[df['unit'].isin(fold_val_units)].reset_index(drop=True)

    (fold_train_df, fold_val_df), fold_feature_cols, _, _, _, _ = engineer_features(fold_train_raw, fold_val_raw)

    X_tr = fold_train_df[fold_feature_cols]
    y_tr = fold_train_df['RUL']
    X_va = fold_val_df[fold_feature_cols]
    y_va = fold_val_df['RUL']

    fold_model = XGBRegressor(**search.best_params_, random_state=42, n_jobs=-1)
    fold_model.fit(X_tr, y_tr)
    fold_preds = fold_model.predict(X_va)

    fold_rmse = np.sqrt(mean_squared_error(y_va, fold_preds))
    fold_r2 = r2_score(y_va, fold_preds)
    rmse_scores.append(fold_rmse)
    r2_scores.append(fold_r2)
    print(f"Fold {fold + 1}: RMSE={fold_rmse:.2f}, R²={fold_r2:.4f}")

print(f"\nAverage RMSE: {np.mean(rmse_scores):.2f} (+/- {np.std(rmse_scores):.2f})")
print(f"Average R²: {np.mean(r2_scores):.4f} (+/- {np.std(r2_scores):.2f})")

scaler = StandardScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
val_df[feature_cols] = scaler.transform(val_df[feature_cols])


def make_windows(data, window, cols):
    X, y = [], []
    for unit_id, g in data.groupby('unit'):
        g = g.sort_values('cycle')
        values = g[cols].values
        ruls = g['RUL'].values

        if len(g) < window:
            pad_len = window - len(g)
            pad = np.repeat(values[0:1], pad_len, axis=0)
            values = np.vstack([pad, values])
            ruls = np.concatenate([np.full(pad_len, ruls[0]), ruls])

        for i in range(len(values) - window + 1):
            X.append(values[i:i + window])
            y.append(ruls[i + window - 1])

    return np.array(X), np.array(y)


def build_lstm(window, n_features, units1, units2, dropout, lr):
    from tensorflow.keras.optimizers import Adam
    m = Sequential([
        LSTM(units1, return_sequences=True, input_shape=(window, n_features)),
        Dropout(dropout),
        LSTM(units2),
        Dropout(dropout),
        Dense(1)
    ])
    m.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])
    return m


candidates = [
    {'window': 30, 'units1': 64, 'units2': 32, 'dropout': 0.2, 'lr': 1e-3},
    {'window': 30, 'units1': 128, 'units2': 64, 'dropout': 0.3, 'lr': 1e-3},
    {'window': 20, 'units1': 64, 'units2': 32, 'dropout': 0.2, 'lr': 1e-3},
    {'window': 45, 'units1': 64, 'units2': 32, 'dropout': 0.2, 'lr': 5e-4},
]

search_results = []
for cand in candidates:
    Xw_train, yw_train = make_windows(train_df, cand['window'], feature_cols)
    Xw_val, yw_val = make_windows(val_df, cand['window'], feature_cols)

    m = build_lstm(cand['window'], Xw_train.shape[2], cand['units1'], cand['units2'], cand['dropout'], cand['lr'])
    m.fit(
        Xw_train, yw_train,
        validation_data=(Xw_val, yw_val),
        epochs=25,
        batch_size=64,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
        verbose=0,
    )
    val_preds = m.predict(Xw_val, verbose=0).flatten()
    cand_rmse = np.sqrt(mean_squared_error(yw_val, val_preds))
    search_results.append(cand_rmse)
    print(f"LSTM candidate {cand}: quick-search val RMSE={cand_rmse:.2f}")

best_idx = int(np.argmin(search_results))
best_cand = candidates[best_idx]
print(f"\nBest LSTM candidate: {best_cand} (quick-search RMSE={search_results[best_idx]:.2f})")

WINDOW = best_cand['window']
X_train, y_train = make_windows(train_df, WINDOW, feature_cols)
X_val, y_val = make_windows(val_df, WINDOW, feature_cols)

lstm_model = build_lstm(WINDOW, X_train.shape[2], best_cand['units1'], best_cand['units2'], best_cand['dropout'], best_cand['lr'])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)

history = lstm_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

lstm_preds = lstm_model.predict(X_val).flatten()
lstm_rmse = np.sqrt(mean_squared_error(y_val, lstm_preds))
lstm_r2 = r2_score(y_val, lstm_preds)
print(f"LSTM Validation RMSE: {lstm_rmse:.2f} cycles")
print(f"LSTM Validation R²: {lstm_r2:.4f}")

# --- persist trained models + preprocessing for later test-set evaluation ---
import joblib
from pathlib import Path

ARTIFACT_DIR = Path("models/artifacts/FD003")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(model, ARTIFACT_DIR / "tree_model.joblib")
lstm_model.save(ARTIFACT_DIR / "lstm_model.keras")

joblib.dump({
    'sensor_cols': SENSOR_COLS,
    'roll_window': ROLL_WINDOW,
    'trend_scaler': fitted['trend_scaler'],
    'fault_kmeans': fitted['fault_kmeans'],
    'informative_sensors': informative_sensors,
    'near_constant': near_constant,
    'feature_cols': feature_cols,
    'lstm_scaler': scaler,
    'window': WINDOW,
    'cap': CAP,
}, ARTIFACT_DIR / "preprocessing.joblib")

print(f"\nSaved trained models and preprocessing to {ARTIFACT_DIR}/")
