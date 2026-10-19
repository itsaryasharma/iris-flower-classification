"""Prediction helpers used by the Streamlit app and the tests.

Nothing in this file depends on Streamlit, so it can be tested directly.
"""
import json
import math

import joblib
import numpy as np
import pandas as pd

from src.config import FEATURE_LABELS, FEATURES, MAX_CM, METRICS_PATH, MIN_CM, MODEL_PATH, SPECIES


def ensure_trained() -> None:
    """Train the model if the saved artifacts are missing (e.g. fresh clone)."""
    if not (MODEL_PATH.exists() and METRICS_PATH.exists()):
        from src.train import main as train_main
        train_main()


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def load_metrics(path=METRICS_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_measurements(values: dict) -> list[str]:
    """Return a list of error messages; an empty list means the input is usable."""
    errors = []
    for f in FEATURES:
        label = FEATURE_LABELS[f]
        if f not in values:
            errors.append(f"{label} is missing.")
            continue
        v = values[f]
        try:
            v = float(v)
        except (TypeError, ValueError):
            errors.append(f"{label} must be a number.")
            continue
        if math.isnan(v) or math.isinf(v):
            errors.append(f"{label} must be a finite number.")
        elif v <= MIN_CM:
            errors.append(f"{label} must be greater than {MIN_CM:g} cm.")
        elif v > MAX_CM:
            errors.append(f"{label} must be at most {MAX_CM:g} cm.")
    return errors


def out_of_range_warnings(values: dict, train_min: dict, train_max: dict) -> list[str]:
    """Warn when a value lies outside the range seen during training.

    The model will still return an answer, but it is extrapolating, so the
    answer deserves less trust.
    """
    warnings = []
    for f in FEATURES:
        v = float(values[f])
        if v < train_min[f] or v > train_max[f]:
            warnings.append(
                f"{FEATURE_LABELS[f]} = {v:g} cm is outside the training range "
                f"({train_min[f]:g}–{train_max[f]:g} cm)."
            )
    return warnings


def to_frame(values: dict) -> pd.DataFrame:
    return pd.DataFrame([[float(values[f]) for f in FEATURES]], columns=FEATURES)


def predict_species(model, values: dict) -> dict:
    """Predict a species and per-class probabilities for one flower."""
    errors = validate_measurements(values)
    if errors:
        raise ValueError(" ".join(errors))
    proba = model.predict_proba(to_frame(values))[0]
    best = int(np.argmax(proba))
    return {
        "species": SPECIES[best],
        "confidence": float(proba[best]),
        "probabilities": {sp: float(p) for sp, p in zip(SPECIES, proba)},
    }


def decision_grid(model, sepal_length: float, sepal_width: float,
                  x_range=(0.5, 7.5), y_range=(0.0, 2.8), steps: int = 150):
    """Predicted class over a petal-length x petal-width grid.

    The model uses all four features, so a 2-D picture is a *slice*: the two
    sepal measurements are held fixed at the user's values.
    Returns (xs, ys, zz) where zz[i, j] is the class index at (xs[j], ys[i]).
    """
    xs = np.linspace(*x_range, steps)
    ys = np.linspace(*y_range, steps)
    gx, gy = np.meshgrid(xs, ys)
    grid = pd.DataFrame({
        "sepal_length": np.full(gx.size, sepal_length),
        "sepal_width": np.full(gx.size, sepal_width),
        "petal_length": gx.ravel(),
        "petal_width": gy.ravel(),
    })[FEATURES]
    zz = model.predict(grid).reshape(gx.shape)
    return xs, ys, zz
