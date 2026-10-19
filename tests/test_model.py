"""Tests for the data, training pipeline and prediction helpers."""
import math

import pytest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.config import FEATURES, RANDOM_STATE, SPECIES, TEST_SIZE
from src.data import load_iris_frame
from src.predict import (
    decision_grid,
    ensure_trained,
    load_metrics,
    load_model,
    out_of_range_warnings,
    predict_species,
    validate_measurements,
)
from src.train import select_model


@pytest.fixture(scope="module")
def model():
    ensure_trained()
    return load_model()


@pytest.fixture(scope="module")
def metrics():
    ensure_trained()
    return load_metrics()


def test_dataset_shape_and_balance():
    df = load_iris_frame()
    assert df.shape == (150, 6)
    assert list(df.columns[:4]) == FEATURES
    assert df[FEATURES].isna().sum().sum() == 0
    assert df["species"].value_counts().to_dict() == {s: 50 for s in SPECIES}


def test_saved_metrics_match_saved_model(model, metrics):
    """The accuracy shown in the app must be reproducible from the saved model."""
    df = load_iris_frame()
    _, X_test, _, y_test = train_test_split(
        df[FEATURES], df["target"], test_size=TEST_SIZE, stratify=df["target"], random_state=RANDOM_STATE
    )
    acc = accuracy_score(y_test, model.predict(X_test))
    assert math.isclose(acc, metrics["test"]["accuracy"])
    assert sum(map(sum, metrics["test"]["confusion_matrix"])) == len(X_test)


def test_model_quality_floor(metrics):
    assert metrics["test"]["accuracy"] >= 0.9
    assert metrics["selected_model"] in metrics["cv_results"]


@pytest.mark.parametrize("values, expected", [
    ({"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}, "setosa"),
    ({"sepal_length": 7.0, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4}, "versicolor"),
    ({"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5}, "virginica"),
])
def test_known_examples(model, values, expected):
    result = predict_species(model, values)
    assert result["species"] == expected
    assert math.isclose(sum(result["probabilities"].values()), 1.0, abs_tol=1e-9)
    assert 0 <= result["confidence"] <= 1


@pytest.mark.parametrize("values", [
    {"sepal_length": -1, "sepal_width": 3, "petal_length": 1, "petal_width": 0.2},   # negative
    {"sepal_length": 0, "sepal_width": 3, "petal_length": 1, "petal_width": 0.2},    # zero
    {"sepal_length": 5, "sepal_width": 3, "petal_length": 99, "petal_width": 0.2},   # absurd
    {"sepal_length": "abc", "sepal_width": 3, "petal_length": 1, "petal_width": 0.2},
    {"sepal_length": float("nan"), "sepal_width": 3, "petal_length": 1, "petal_width": 0.2},
    {"sepal_width": 3, "petal_length": 1, "petal_width": 0.2},                       # missing
])
def test_invalid_inputs_are_rejected(model, values):
    assert validate_measurements(values)
    with pytest.raises(ValueError):
        predict_species(model, values)


def test_out_of_range_warning(metrics):
    stats = metrics["train_stats"]
    inside = {"sepal_length": 5.8, "sepal_width": 3.0, "petal_length": 4.3, "petal_width": 1.3}
    outside = dict(inside, petal_length=9.5)
    assert out_of_range_warnings(inside, stats["min"], stats["max"]) == []
    warnings = out_of_range_warnings(outside, stats["min"], stats["max"])
    assert len(warnings) == 1 and "Petal length" in warnings[0]


def test_decision_grid_shape(model):
    xs, ys, zz = decision_grid(model, 5.8, 3.0, steps=20)
    assert zz.shape == (20, 20)
    assert set(zz.ravel()) <= {0, 1, 2}


def test_selection_prefers_simpler_model_on_near_tie():
    cv = {"Simple": {"mean": 0.955}, "Complex": {"mean": 0.960}}
    assert select_model(cv) == "Simple"
    cv = {"Simple": {"mean": 0.90}, "Complex": {"mean": 0.96}}
    assert select_model(cv) == "Complex"
