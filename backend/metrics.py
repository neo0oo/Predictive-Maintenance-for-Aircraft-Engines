import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2(y_true, y_pred):
    return float(r2_score(y_true, y_pred))


def nasa_score(y_true, y_pred):
    # PHM08 challenge scoring: asymmetric exponential penalty, harsher on
    # late predictions (predicted RUL > true RUL) than early ones. Summed over
    # engines; lower is better.
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def summarize(y_true, y_pred):
    return {
        "rmse": rmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred),
        "n_engines": int(len(y_true)),
    }
