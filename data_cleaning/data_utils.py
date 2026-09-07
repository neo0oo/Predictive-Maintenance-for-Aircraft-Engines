import pandas as pd
from pathlib import Path

# Anchored to the project root (parent of this file's folder) rather than a
# relative path, so it works regardless of the notebook's working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = RAW_DIR / "processed"

FD_DATASETS = ["FD001", "FD002", "FD003", "FD004"]

# Column names per the readme: unit id, cycle, 3 op settings, 21 sensors
COLUMNS = (
    ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)
FEATURE_COLUMNS = ["op_setting_1", "op_setting_2", "op_setting_3"] + [
    f"sensor_{i}" for i in range(1, 22)
]


def load_cmapss_file(path):
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)


def load_rul_file(path):
    return pd.read_csv(path, sep=r"\s+", header=None, names=["RUL"])


def load_dataset(fd, data_dir=RAW_DIR):
    fd_dir = data_dir / fd
    return {
        "train": load_cmapss_file(fd_dir / f"train_{fd}.txt"),
        "test": load_cmapss_file(fd_dir / f"test_{fd}.txt"),
        "rul": load_rul_file(fd_dir / f"RUL_{fd}.txt"),
    }


def load_all_datasets(data_dir=RAW_DIR):
    return {fd: load_dataset(fd, data_dir) for fd in FD_DATASETS}


def add_train_rul(df):
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df = df.copy()
    df["RUL"] = max_cycle - df["cycle"]
    return df
