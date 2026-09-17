import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .explain import Explainers
from .metrics import summarize
from .pipelines import (
    ARTIFACTS,
    CRITICAL_BELOW,
    DATASET_INFO,
    DATASETS,
    DEGRADING_BELOW,
    SENSOR_NAMES,
    Pipeline,
    health_status,
    load_test_set,
    load_train_set,
    parse_uploaded_trajectory,
)

MODEL_LABELS = {"random_forest": "Random Forest", "xgboost": "XGBoost", "lstm": "LSTM"}
SEED_PATH = Path(__file__).resolve().parent / "validation_seed.json"


class DatasetState:
    def __init__(self, fd):
        self.fd = fd
        self.pipeline = Pipeline(fd)
        p = self.pipeline

        self.test_raw, self.rul_last = load_test_set(fd)
        self.test_eng = p.engineer(self.test_raw)
        self.test_lstm = p.lstm_frame(self.test_eng)
        self.n_cycles = self.test_eng.groupby("unit")["cycle"].max()
        self.true_rul_capped = np.clip(self.rul_last, a_min=None, a_max=p.cap)

        last_rows = self.test_eng.sort_values("cycle").groupby("unit").tail(1).sort_values("unit")
        tree_preds = p.predict_tree(last_rows)

        lstm_windows = [
            p.window_ending_at(self.test_lstm[self.test_lstm["unit"] == u], int(self.n_cycles[u]))
            for u in last_rows["unit"]
        ]
        lstm_preds = p.predict_lstm_windows(lstm_windows)

        self.last_predictions = {p.tree_key: tree_preds, "lstm": lstm_preds}
        self.test_metrics = {
            key: summarize(self.true_rul_capped, preds) for key, preds in self.last_predictions.items()
        }

        train = load_train_set(fd)
        self.sensor_ranges = self._ranges(train, p.sensors)
        op_cols = [c for c in p.required_raw_columns() if c.startswith("op_setting")]
        self.op_ranges = self._ranges(train, op_cols) if op_cols else {}
        self.cycle_range = {"min": 1, "max": int(train["cycle"].max()), "median": int(train["cycle"].median())}

        # in the multi-condition datasets a sensor's plausible range depends heavily on
        # which regime the engine is in, so sliders need per-regime stats plus each
        # regime's op-setting centroid rather than one global range
        self.regimes = None
        if p.has_regimes:
            kmeans = p.prep["kmeans"]
            regime_ids = kmeans.predict(train[p.prep["op_cols"]])
            self.regimes = []
            for r in range(p.prep["n_regimes"]):
                subset = train[regime_ids == r]
                self.regimes.append({
                    "id": r,
                    "n_train_rows": int(len(subset)),
                    "op_settings": {c: float(v) for c, v in zip(p.prep["op_cols"], kmeans.cluster_centers_[r])},
                    "sensors": self._ranges(subset, p.sensors),
                })

        self.majority_fault_cluster = None
        if p.has_fault_clusters:
            train_eng = p.engineer(train)
            per_unit = train_eng.groupby("unit")["fault_cluster"].first()
            self.majority_fault_cluster = int(per_unit.value_counts().idxmax())

        self.validation = self._load_validation()

    @staticmethod
    def _ranges(df, cols):
        out = {}
        for c in cols:
            s = df[c].astype(float)
            out[c] = {
                "min": float(s.min()),
                "max": float(s.max()),
                "p5": float(s.quantile(0.05)),
                "p95": float(s.quantile(0.95)),
                "median": float(s.median()),
            }
        return out

    def _load_validation(self):
        metrics_path = ARTIFACTS / self.fd / "metrics.json"
        if metrics_path.exists():
            return json.loads(metrics_path.read_text())
        return json.loads(SEED_PATH.read_text()).get(self.fd, {})

    def unit_frames(self, unit):
        eng = self.test_eng[self.test_eng["unit"] == unit]
        if eng.empty:
            raise HTTPException(404, f"engine {unit} not in {self.fd} test set")
        return eng, self.test_lstm[self.test_lstm["unit"] == unit]


STATE = {}
EXPLAINERS = Explainers()


@asynccontextmanager
async def lifespan(app):
    for fd in DATASETS:
        STATE[fd] = DatasetState(fd)
    yield


app = FastAPI(title="Turbofan RUL API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_state(fd):
    if fd not in STATE:
        raise HTTPException(404, f"unknown dataset {fd}")
    return STATE[fd]


def require_model(state, model):
    if model not in state.pipeline.available_models:
        raise HTTPException(
            400, f"{MODEL_LABELS.get(model, model)} is not trained for {state.fd}; "
                 f"available: {state.pipeline.available_models}"
        )


def sensor_meta(key):
    short, desc = SENSOR_NAMES[key]
    return {"key": key, "name": short, "description": desc}


def explain_for(state, model, tree_row=None, window=None):
    if model == "lstm":
        return EXPLAINERS.lstm(state.pipeline, window)
    return EXPLAINERS.tree(state.pipeline, tree_row)


@app.get("/api/datasets")
def datasets():
    return [
        {
            "id": fd,
            "label": DATASET_INFO[fd]["label"],
            "conditions": DATASET_INFO[fd]["conditions"],
            "fault_modes": DATASET_INFO[fd]["fault_modes"],
            "available_models": STATE[fd].pipeline.available_models,
            "n_test_engines": int(len(STATE[fd].rul_last)),
            "sensors": [sensor_meta(s) for s in STATE[fd].pipeline.sensors],
            "lstm_window": STATE[fd].pipeline.window,
        }
        for fd in DATASETS
    ]


@app.get("/api/thresholds")
def thresholds():
    return {"critical_below": CRITICAL_BELOW, "degrading_below": DEGRADING_BELOW}


@app.get("/api/fleet/{fd}/{model}")
def fleet(fd: str, model: str):
    state = get_state(fd)
    require_model(state, model)
    preds = state.last_predictions[model]
    engines = []
    counts = {"healthy": 0, "degrading": 0, "critical": 0}
    for i, unit in enumerate(range(1, len(preds) + 1)):
        status = health_status(float(preds[i]))
        counts[status] += 1
        engines.append({
            "unit": unit,
            "n_cycles": int(state.n_cycles[unit]),
            "predicted_rul": float(preds[i]),
            "true_rul": float(state.true_rul_capped[i]),
            "status": status,
        })
    return {
        "dataset": fd,
        "model": model,
        "metrics": state.test_metrics[model],
        "status_counts": counts,
        "engines": engines,
    }


@app.get("/api/engine/{fd}/{unit}")
def engine(fd: str, unit: int):
    state = get_state(fd)
    eng, _ = state.unit_frames(unit)
    p = state.pipeline
    raw = state.test_raw[state.test_raw["unit"] == unit].sort_values("cycle")
    cols = ["cycle"] + p.sensors + [c for c in ["op_setting_1", "op_setting_2", "op_setting_3"] if c in raw.columns]
    history = raw[cols].to_dict(orient="records")
    n_cycles = int(state.n_cycles[unit])
    return {
        "dataset": fd,
        "unit": unit,
        "n_cycles": n_cycles,
        "true_rul_at_last_cycle": float(state.rul_last[unit - 1]),
        "available_models": p.available_models,
        "sensors": [sensor_meta(s) for s in p.sensors],
        "history": history,
        "regime_per_cycle": eng.sort_values("cycle")["regime"].tolist() if p.has_regimes else None,
        "fault_cluster": int(eng["fault_cluster"].iloc[0]) if p.has_fault_clusters else None,
    }


@app.get("/api/engine/{fd}/{unit}/predict")
def engine_predict(fd: str, unit: int, model: str, cycle: int):
    state = get_state(fd)
    require_model(state, model)
    eng, lstm_eng = state.unit_frames(unit)
    p = state.pipeline
    n_cycles = int(state.n_cycles[unit])
    if cycle < 1 or cycle > n_cycles:
        raise HTTPException(400, f"cycle must be in 1..{n_cycles}")

    if model == "lstm":
        window = p.window_ending_at(lstm_eng, cycle)
        pred = float(p.predict_lstm_windows([window])[0])
        explanation = explain_for(state, model, window=window)
    else:
        row = eng[eng["cycle"] == cycle]
        pred = float(p.predict_tree(row)[0])
        explanation = explain_for(state, model, tree_row=row)

    true_rul = float(state.rul_last[unit - 1]) + (n_cycles - cycle)
    return {
        "dataset": fd,
        "unit": unit,
        "model": model,
        "cycle": cycle,
        "predicted_rul": pred,
        "true_rul": true_rul,
        "true_rul_capped": float(min(true_rul, p.cap)),
        "status": health_status(pred),
        "explanation": explanation,
    }


@app.get("/api/comparison")
def comparison():
    out = {}
    for fd in DATASETS:
        state = STATE[fd]
        models = {}
        for key in state.pipeline.available_models:
            val = state.validation.get(key, {})
            models[key] = {
                "label": MODEL_LABELS[key],
                "test": state.test_metrics[key],
                "validation": val.get("validation"),
                "cv": val.get("cv"),
            }
        out[fd] = {"label": DATASET_INFO[fd]["label"], "models": models}
    return out


@app.get("/api/sensor-ranges/{fd}")
def sensor_ranges(fd: str):
    state = get_state(fd)
    p = state.pipeline
    return {
        "dataset": fd,
        "sensors": [dict(sensor_meta(s), **state.sensor_ranges[s]) for s in p.sensors],
        "op_settings": [dict(key=k, **v) for k, v in state.op_ranges.items()],
        "regimes": state.regimes,
        "cycle": state.cycle_range if fd == "FD001" else None,
        "fault_cluster": (
            {"default": state.majority_fault_cluster, "options": [0, 1]} if p.has_fault_clusters else None
        ),
        "lstm_window": p.window,
        "available_models": p.available_models,
    }


class ManualRequest(BaseModel):
    dataset: str
    model: str
    sensors: dict[str, float]
    op_settings: dict[str, float] = {}
    cycle: Optional[int] = None
    fault_cluster: Optional[int] = None


def steady_state_trajectory(state, req):
    # a single sensor snapshot has no history, so it's treated as an engine that has
    # been holding these exact readings for a full LSTM window: rolling std becomes
    # 0, rolling mean equals the reading, and the window is the snapshot repeated
    p = state.pipeline
    length = p.window
    last_cycle = int(req.cycle) if req.cycle is not None else state.cycle_range["median"]
    last_cycle = max(1, last_cycle)
    cycles = np.arange(max(1, last_cycle - length + 1), last_cycle + 1)
    row = {"unit": 0}
    for s in p.sensors:
        if s not in req.sensors:
            raise HTTPException(400, f"missing sensor value for {s}")
        row[s] = float(req.sensors[s])
    for col in p.required_raw_columns():
        if col.startswith("op_setting"):
            if col not in req.op_settings:
                raise HTTPException(400, f"missing op setting {col}")
            row[col] = float(req.op_settings[col])
    df = pd.DataFrame([dict(row, cycle=int(c)) for c in cycles])
    return df, last_cycle


@app.post("/api/predict/manual")
def predict_manual(req: ManualRequest):
    state = get_state(req.dataset)
    require_model(state, req.model)
    p = state.pipeline
    df, last_cycle = steady_state_trajectory(state, req)

    override = None
    if p.has_fault_clusters:
        override = req.fault_cluster if req.fault_cluster is not None else state.majority_fault_cluster
    eng = p.engineer(df, fault_cluster_override=override)

    if req.model == "lstm":
        lstm_eng = p.lstm_frame(eng)
        window = p.window_ending_at(lstm_eng, last_cycle)
        pred = float(p.predict_lstm_windows([window])[0])
        explanation = explain_for(state, req.model, window=window)
    else:
        row = eng[eng["cycle"] == last_cycle]
        pred = float(p.predict_tree(row)[0])
        explanation = explain_for(state, req.model, tree_row=row)

    return {
        "dataset": req.dataset,
        "model": req.model,
        "predicted_rul": pred,
        "status": health_status(pred),
        "assumptions": {
            "steady_state": True,
            "cycle": last_cycle,
            "regime": int(eng["regime"].iloc[-1]) if p.has_regimes else None,
            "fault_cluster": override,
        },
        "explanation": explanation,
    }


@app.post("/api/predict/csv")
async def predict_csv(dataset: str, model: str, file: UploadFile = File(...)):
    state = get_state(dataset)
    require_model(state, model)
    p = state.pipeline
    try:
        raw = parse_uploaded_trajectory(await file.read())
        if "unit" not in raw.columns:
            raw["unit"] = 1
        eng = p.engineer(raw)
    except ValueError as e:
        raise HTTPException(400, str(e))

    lstm_eng = p.lstm_frame(eng)
    results = []
    for unit, g in eng.groupby("unit"):
        g = g.sort_values("cycle")
        tree_series = p.predict_tree(g).tolist()
        cycles, windows = p.all_windows_for_unit(lstm_eng[lstm_eng["unit"] == unit])
        lstm_series = p.predict_lstm_windows(windows).tolist()
        last_cycle = int(g["cycle"].iloc[-1])
        if model == "lstm":
            explanation = explain_for(state, model, window=windows[-1])
            final = lstm_series[-1]
        else:
            explanation = explain_for(state, model, tree_row=g.tail(1))
            final = tree_series[-1]
        results.append({
            "unit": int(unit),
            "n_cycles": len(g),
            "cycles": [int(c) for c in cycles],
            "predictions": {p.tree_key: tree_series, "lstm": lstm_series},
            "final_predicted_rul": float(final),
            "final_cycle": last_cycle,
            "status": health_status(float(final)),
            "explanation": explanation,
        })
    return {"dataset": dataset, "model": model, "engines": results}
