import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = PROJECT_ROOT / "models" / "artifacts"
DATA = PROJECT_ROOT / "data" / "processed"

DATASETS = ["FD001", "FD002", "FD003", "FD004"]

DATASET_INFO = {
    "FD001": {"conditions": 1, "fault_modes": 1, "label": "1 operating condition, 1 fault mode"},
    "FD002": {"conditions": 6, "fault_modes": 1, "label": "6 operating conditions, 1 fault mode"},
    "FD003": {"conditions": 1, "fault_modes": 2, "label": "1 operating condition, 2 fault modes"},
    "FD004": {"conditions": 6, "fault_modes": 2, "label": "6 operating conditions, 2 fault modes"},
}

SENSOR_NAMES = {
    "sensor_1": ("T2", "Fan inlet temp"),
    "sensor_2": ("T24", "LPC outlet temp"),
    "sensor_3": ("T30", "HPC outlet temp"),
    "sensor_4": ("T50", "LPT outlet temp"),
    "sensor_5": ("P2", "Fan inlet pressure"),
    "sensor_6": ("P15", "Bypass duct pressure"),
    "sensor_7": ("P30", "HPC outlet pressure"),
    "sensor_8": ("Nf", "Fan speed"),
    "sensor_9": ("Nc", "Core speed"),
    "sensor_10": ("epr", "Engine pressure ratio"),
    "sensor_11": ("Ps30", "HPC static pressure"),
    "sensor_12": ("phi", "Fuel flow ratio"),
    "sensor_13": ("NRf", "Corrected fan speed"),
    "sensor_14": ("NRc", "Corrected core speed"),
    "sensor_15": ("BPR", "Bypass ratio"),
    "sensor_16": ("farB", "Burner fuel-air ratio"),
    "sensor_17": ("htBleed", "Bleed enthalpy"),
    "sensor_18": ("Nf_dmd", "Demanded fan speed"),
    "sensor_19": ("PCNfR_dmd", "Demanded corrected fan speed"),
    "sensor_20": ("W31", "HPT coolant bleed"),
    "sensor_21": ("W32", "LPT coolant bleed"),
}

RAW_COLUMNS = (
    ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

CRITICAL_BELOW = 30
DEGRADING_BELOW = 70


def health_status(rul):
    if rul < CRITICAL_BELOW:
        return "critical"
    if rul < DEGRADING_BELOW:
        return "degrading"
    return "healthy"


def compute_engine_trends(data, sensor_cols):
    rows = []
    for unit_id, g in data.groupby("unit"):
        g = g.sort_values("cycle")
        slopes = {c: np.polyfit(g["cycle"], g[c], 1)[0] for c in sensor_cols}
        slopes["unit"] = unit_id
        rows.append(slopes)
    return pd.DataFrame(rows).set_index("unit")


def add_rolling(df, sensor_cols, window):
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    grouped = df.groupby("unit")[sensor_cols]
    roll_mean = grouped.rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
    roll_std = grouped.rolling(window=window, min_periods=1).std().fillna(0).reset_index(level=0, drop=True)
    roll_mean.columns = [f"{c}_rollmean" for c in sensor_cols]
    roll_std.columns = [f"{c}_rollstd" for c in sensor_cols]
    return pd.concat([df, roll_mean, roll_std], axis=1)


def add_regime_features(df, n_regimes):
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    regime_changed = df.groupby("unit")["regime"].diff().fillna(1) != 0
    streak_id = regime_changed.groupby(df["unit"]).cumsum()
    df["regime_dwell"] = df.groupby([df["unit"], streak_id]).cumcount() + 1
    for r in range(n_regimes):
        df[f"regime_{r}"] = (df["regime"] == r).astype(int)
    return df


class Pipeline:
    """Transform-only reconstruction of one dataset's training preprocessing,
    driven entirely by the fitted objects saved in models/artifacts/<fd>/."""

    def __init__(self, fd):
        self.fd = fd
        art = ARTIFACTS / fd
        self.prep = joblib.load(art / "preprocessing.joblib")
        self.tree_model = joblib.load(art / "tree_model.joblib")
        self.lstm_model = load_model(art / "lstm_model.keras")
        self.tree_key = "random_forest" if fd == "FD001" else "xgboost"
        self.cap = self.prep["cap"]
        self.window = self.prep["window"]
        self.roll_window = self.prep["roll_window"]
        self.has_regimes = "kmeans" in self.prep
        self.has_fault_clusters = "fault_kmeans" in self.prep

        if fd == "FD001":
            self.sensors = list(self.prep["sensor_cols"])
            self.tree_cols = list(self.prep["tree_feature_cols"])
            self.lstm_cols = list(self.prep["lstm_feature_cols"])
        else:
            self.sensors = list(self.prep["informative_sensors"])
            self.tree_cols = list(self.prep["feature_cols"])
            self.lstm_cols = list(self.prep["feature_cols"])
        self.lstm_scaler = self.prep.get("lstm_scaler")

    @property
    def available_models(self):
        return [self.tree_key, "lstm"]

    def required_raw_columns(self):
        cols = ["unit", "cycle"]
        if self.has_regimes:
            cols += list(self.prep["op_cols"])
        elif self.fd == "FD001":
            cols += ["op_setting_1", "op_setting_2"]
        cols += self.sensors
        return cols

    def engineer(self, raw, fault_cluster_override=None):
        df = raw.copy()
        missing = [c for c in self.required_raw_columns() if c not in df.columns]
        if missing:
            raise ValueError(f"missing columns for {self.fd}: {missing}")

        if self.fd == "FD001":
            return add_rolling(df, self.sensors, self.roll_window)

        prep = self.prep
        df[self.sensors] = df[self.sensors].astype(float)

        if self.has_regimes:
            df["regime"] = prep["kmeans"].predict(df[prep["op_cols"]])
            for regime, scaler in prep["regime_scalers"].items():
                mask = df["regime"] == regime
                if mask.any():
                    df.loc[mask, self.sensors] = scaler.transform(df.loc[mask, self.sensors])
            df = add_regime_features(df, prep["n_regimes"])

        if self.has_fault_clusters:
            if fault_cluster_override is None:
                trends = compute_engine_trends(df, self.sensors)
                labels = prep["fault_kmeans"].predict(prep["trend_scaler"].transform(trends))
                cluster_map = pd.Series(labels, index=trends.index)
                df["fault_cluster"] = df["unit"].map(cluster_map)
            else:
                df["fault_cluster"] = int(fault_cluster_override)
            df["fault_cluster_0"] = (df["fault_cluster"] == 0).astype(int)
            df["fault_cluster_1"] = (df["fault_cluster"] == 1).astype(int)

        return add_rolling(df, self.sensors, self.roll_window)

    def lstm_frame(self, eng):
        out = eng.copy()
        if self.lstm_scaler is not None:
            out[self.lstm_cols] = self.lstm_scaler.transform(out[self.lstm_cols])
        return out

    def predict_tree(self, eng):
        return self.tree_model.predict(eng[self.tree_cols])

    def window_ending_at(self, lstm_eng_unit, cycle):
        g = lstm_eng_unit[lstm_eng_unit["cycle"] <= cycle].sort_values("cycle")
        values = g[self.lstm_cols].values
        if len(values) == 0:
            raise ValueError("no cycles at or before requested cycle")
        if len(values) < self.window:
            pad = np.repeat(values[0:1], self.window - len(values), axis=0)
            values = np.vstack([pad, values])
        return values[-self.window:]

    def predict_lstm_windows(self, windows):
        x = np.asarray(windows, dtype=np.float32)
        return np.asarray(self.lstm_model(x, training=False)).reshape(-1)

    def all_windows_for_unit(self, lstm_eng_unit):
        g = lstm_eng_unit.sort_values("cycle")
        values = g[self.lstm_cols].values
        cycles = g["cycle"].values
        if len(values) < self.window:
            pad = np.repeat(values[0:1], self.window - len(values), axis=0)
            values = np.vstack([pad, values])
        windows = [values[max(0, i - self.window + 1): i + 1] for i in range(len(values))]
        windows = [
            np.vstack([np.repeat(w[0:1], self.window - len(w), axis=0), w]) if len(w) < self.window else w
            for w in windows
        ]
        windows = windows[-len(cycles):]
        return cycles, np.array(windows)


def load_test_set(fd):
    test = pd.read_csv(DATA / f"test_{fd}_clean.csv")
    rul = pd.read_csv(DATA / f"RUL_{fd}_clean.csv")
    units = sorted(test["unit"].unique())
    assert units == list(range(1, len(units) + 1))
    return test, rul["RUL"].values


def load_train_set(fd):
    return pd.read_csv(DATA / f"train_{fd}_clean.csv")


def parse_uploaded_trajectory(raw_bytes):
    text = raw_bytes.decode("utf-8", errors="replace")
    first = text.splitlines()[0] if text.strip() else ""
    from io import StringIO
    if "unit" in first and "cycle" in first:
        return pd.read_csv(StringIO(text))
    df = pd.read_csv(StringIO(text), sep=r"\s+", header=None)
    if df.shape[1] == len(RAW_COLUMNS):
        df.columns = RAW_COLUMNS
    elif df.shape[1] == len(RAW_COLUMNS) + 1 and df.iloc[:, -1].isna().all():
        df = df.iloc[:, :-1]
        df.columns = RAW_COLUMNS
    else:
        raise ValueError(
            f"expected a header row naming unit/cycle/op_setting/sensor columns, or the raw "
            f"26-column space-separated C-MAPSS format; got {df.shape[1]} columns"
        )
    return df
