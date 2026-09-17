import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold, KFold, RandomizedSearchCV
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

df = pd.read_csv("data/processed/train_FD004_clean.csv")

CAP = 125
df['RUL'] = df['RUL'].clip(upper=CAP)

OP_COLS = ['op_setting_1', 'op_setting_2', 'op_setting_3']
SENSOR_COLS = [c for c in df.columns if c.startswith('sensor_')]
N_REGIMES = 6
FLAT_CV_THRESHOLD = 1e-4
ROLL_WINDOW = 5


def compute_engine_trends(data, sensor_cols):
    rows = []
    for unit_id, g in data.groupby('unit'):
        g = g.sort_values('cycle')
        slopes = {c: np.polyfit(g['cycle'], g[c], 1)[0] for c in sensor_cols}
        slopes['unit'] = unit_id
        rows.append(slopes)
    return pd.DataFrame(rows).set_index('unit')


def engineer_features(fit_df, *apply_dfs, report_diagnostics=False):
    frames = [fit_df.copy()] + [f.copy() for f in apply_dfs]
    fit = frames[0]

    kmeans = KMeans(n_clusters=N_REGIMES, random_state=42, n_init=10)
    kmeans.fit(fit[OP_COLS])
    for f in frames:
        f['regime'] = kmeans.predict(f[OP_COLS])

    regime_std = fit.groupby('regime')[SENSOR_COLS].std()
    regime_mean = fit.groupby('regime')[SENSOR_COLS].mean()
    regime_cv = (regime_std / regime_mean.abs()).abs()
    near_constant = regime_cv.columns[(regime_cv < FLAT_CV_THRESHOLD).all(axis=0)].tolist()
    informative_sensors = [c for c in SENSOR_COLS if c not in near_constant]

    for f in frames:
        f[informative_sensors] = f[informative_sensors].astype(float)

    if report_diagnostics:
        raw_trends = compute_engine_trends(fit, informative_sensors)
        raw_trends_scaled = StandardScaler().fit_transform(raw_trends)
        raw_labels = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(raw_trends_scaled)
        raw_silhouette = silhouette_score(raw_trends_scaled, raw_labels)
        print(f"fault-mode cluster separation on RAW sensor trends: silhouette={raw_silhouette:.4f}")

    scalers = {}
    for regime in range(N_REGIMES):
        scaler = StandardScaler()
        fit_mask = fit['regime'] == regime
        scaler.fit(fit.loc[fit_mask, informative_sensors])
        scalers[regime] = scaler
        for f in frames:
            mask = f['regime'] == regime
            if mask.any():
                f.loc[mask, informative_sensors] = scaler.transform(f.loc[mask, informative_sensors])

    for f in frames:
        f.sort_values(['unit', 'cycle'], inplace=True)
        f.reset_index(drop=True, inplace=True)

        regime_changed = f.groupby('unit')['regime'].diff().fillna(1) != 0
        streak_id = regime_changed.groupby(f['unit']).cumsum()
        f['regime_dwell'] = f.groupby([f['unit'], streak_id]).cumcount() + 1

        for r in range(N_REGIMES):
            f[f'regime_{r}'] = (f['regime'] == r).astype(int)

    fit_trends = compute_engine_trends(fit, informative_sensors)
    trend_scaler = StandardScaler()
    fit_trends_scaled = trend_scaler.fit_transform(fit_trends)

    fault_kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    fit_labels = fault_kmeans.fit_predict(fit_trends_scaled)
    fit_cluster_map = pd.Series(fit_labels, index=fit_trends.index)

    if report_diagnostics:
        norm_silhouette = silhouette_score(fit_trends_scaled, fit_labels)
        print(f"fault-mode cluster separation on REGIME-NORMALIZED trends: silhouette={norm_silhouette:.4f}")

    cluster_maps = [fit_cluster_map]
    for f in frames[1:]:
        trends = compute_engine_trends(f, informative_sensors)
        labels = fault_kmeans.predict(trend_scaler.transform(trends))
        cluster_maps.append(pd.Series(labels, index=trends.index))

    for f, cmap in zip(frames, cluster_maps):
        f['fault_cluster'] = f['unit'].map(cmap)
        f['fault_cluster_0'] = (f['fault_cluster'] == 0).astype(int)
        f['fault_cluster_1'] = (f['fault_cluster'] == 1).astype(int)

    def add_rolling(f):
        grouped = f.groupby('unit')[informative_sensors]
        roll_mean = grouped.rolling(window=ROLL_WINDOW, min_periods=1).mean().reset_index(level=0, drop=True)
        roll_std = grouped.rolling(window=ROLL_WINDOW, min_periods=1).std().fillna(0).reset_index(level=0, drop=True)
        roll_mean.columns = [f'{c}_rollmean' for c in informative_sensors]
        roll_std.columns = [f'{c}_rollstd' for c in informative_sensors]
        return pd.concat([f, roll_mean, roll_std], axis=1)

    frames = [add_rolling(f) for f in frames]

    feature_cols = (
        [f'regime_{r}' for r in range(N_REGIMES)]
        + ['regime_dwell']
        + ['fault_cluster_0', 'fault_cluster_1']
        + informative_sensors
        + [f'{c}_rollmean' for c in informative_sensors]
        + [f'{c}_rollstd' for c in informative_sensors]
    )

    fitted = {
        'kmeans': kmeans,
        'regime_scalers': scalers,
        'trend_scaler': trend_scaler,
        'fault_kmeans': fault_kmeans,
    }
    return frames, feature_cols, informative_sensors, near_constant, fit_cluster_map, fitted


unit_IDs = df['unit'].unique()
train_units, val_units = train_test_split(unit_IDs, test_size=0.2, random_state=42)

train_raw = df[df['unit'].isin(train_units)].reset_index(drop=True)
val_raw = df[df['unit'].isin(val_units)].reset_index(drop=True)

(train_df, val_df), feature_cols, informative_sensors, near_constant, train_cluster_map, fitted = engineer_features(
    train_raw, val_raw, report_diagnostics=True
)

print("dropping sensors that are flat (regime-aware relative CV):", near_constant)
print(f"informative sensors kept: {len(informative_sensors)}")
print("regime sizes (train):")
print(train_df['regime'].value_counts().sort_index())
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
    n_iter=25,
    cv=GroupKFold(n_splits=3),
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

ARTIFACT_DIR = Path("models/artifacts/FD004")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(model, ARTIFACT_DIR / "tree_model.joblib")
lstm_model.save(ARTIFACT_DIR / "lstm_model.keras")

joblib.dump({
    'op_cols': OP_COLS,
    'sensor_cols': SENSOR_COLS,
    'n_regimes': N_REGIMES,
    'roll_window': ROLL_WINDOW,
    'kmeans': fitted['kmeans'],
    'regime_scalers': fitted['regime_scalers'],
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
