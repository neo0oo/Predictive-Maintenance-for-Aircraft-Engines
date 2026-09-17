import re
import numpy as np
import shap
import tensorflow as tf

from .pipelines import SENSOR_NAMES


def group_for(feature):
    m = re.match(r"(sensor_\d+)", feature)
    if m:
        short, desc = SENSOR_NAMES[m.group(1)]
        return short, desc
    if feature.startswith("regime"):
        return "Regime", "Operating regime (cluster + dwell)"
    if feature.startswith("fault_cluster"):
        return "Fault mode", "Inferred fault-mode cluster"
    if feature == "cycle":
        return "Cycle", "Cycle count"
    if feature.startswith("op_setting"):
        return "Op settings", "Operational settings"
    return feature, feature


def aggregate(feature_names, values):
    # a sensor's raw value, rolling mean and rolling std are three separate model
    # inputs; summing their attributions gives one bar per physical sensor
    agg = {}
    for f, v in zip(feature_names, values):
        label, desc = group_for(f)
        entry = agg.setdefault(label, {"label": label, "description": desc, "value": 0.0})
        entry["value"] += float(v)
    items = list(agg.values())
    for it in items:
        it["abs"] = abs(it["value"])
    items.sort(key=lambda d: -d["abs"])
    total = sum(it["abs"] for it in items) or 1.0
    for it in items:
        it["share"] = it["abs"] / total
    return items


class Explainers:
    def __init__(self):
        self._tree = {}

    def tree(self, pipeline, row_df):
        if pipeline.fd not in self._tree:
            self._tree[pipeline.fd] = shap.TreeExplainer(pipeline.tree_model)
        values = self._tree[pipeline.fd].shap_values(row_df[pipeline.tree_cols])
        values = np.asarray(values).reshape(-1)
        return {"method": "shap", "items": aggregate(pipeline.tree_cols, values)}

    def lstm(self, pipeline, window):
        # gradient x input saliency, summed over the time axis so each feature gets
        # one attribution for the window as a whole
        x = tf.convert_to_tensor(np.asarray(window, dtype=np.float32)[None, ...])
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = pipeline.lstm_model(x, training=False)
        grads = tape.gradient(y, x)
        attr = (grads * x).numpy()[0].sum(axis=0)
        return {"method": "gradient_x_input", "items": aggregate(pipeline.lstm_cols, attr)}
