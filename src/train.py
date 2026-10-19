"""Train, compare and evaluate classifiers, then save the chosen model.

Run from the project root:

    python -m src.train

Steps
-----
1. Load the data and make a stratified 80/20 train/test split.
2. Compare four candidate models with 5-fold stratified cross-validation
   on the TRAINING set only (the test set stays untouched).
3. Pick the model with the best mean CV accuracy. If several are within
   0.01 of the best, prefer the simplest one (order of CANDIDATES).
4. Refit the chosen model on the full training set and evaluate it once on
   the held-out test set.
5. Save the model (joblib) and every reported number (metrics.json) so the
   app displays real results instead of hard-coded ones.
"""
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import sklearn
from scipy.stats import binomtest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    CV_FOLDS,
    FEATURES,
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    SPECIES,
    TEST_SIZE,
)
from src.data import load_iris_frame

SELECTION_TOLERANCE = 0.01


def build_candidates() -> dict:
    """Candidate models, ordered from simplest/most interpretable to least.

    Scale-sensitive models (distance- or margin-based) are wrapped in a
    Pipeline with StandardScaler so that scaling is learned inside each CV
    fold and never sees validation/test data (prevents leakage).
    """
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]),
        "k-Nearest Neighbours (k=5)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Decision Tree (max depth 3)": DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE),
        # SVC has no native probabilities; calibration adds predict_proba
        # (the replacement for the deprecated SVC(probability=True)).
        "SVM (RBF kernel)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(SVC(kernel="rbf"), ensemble=False)),
        ]),
    }


def select_model(cv_results: dict) -> str:
    """Best mean CV accuracy, preferring earlier (simpler) models on near-ties."""
    best = max(r["mean"] for r in cv_results.values())
    for name, r in cv_results.items():  # dicts keep insertion order
        if r["mean"] >= best - SELECTION_TOLERANCE:
            return name
    raise RuntimeError("unreachable")


def accuracy_interval(correct: int, total: int) -> list[float]:
    """95% Wilson score interval for an accuracy of correct/total."""
    ci = binomtest(correct, total).proportion_ci(confidence_level=0.95, method="wilson")
    return [float(ci.low), float(ci.high)]


def train_and_evaluate():
    df = load_iris_frame()
    X, y = df[FEATURES], df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    candidates = build_candidates()
    cv_results = {}
    for name, model in candidates.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        cv_results[name] = {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "folds": [float(s) for s in scores],
        }

    chosen = select_model(cv_results)
    model = candidates[chosen].fit(X_train, y_train)

    y_pred = model.predict(X_test)
    perm = permutation_importance(
        model, X_test, y_test, n_repeats=30, random_state=RANDOM_STATE, scoring="accuracy"
    )

    misclassified = []
    for idx, true, pred in zip(X_test.index, y_test, y_pred):
        if true != pred:
            row = {f: float(X_test.loc[idx, f]) for f in FEATURES}
            row.update(dataset_row=int(idx), true=SPECIES[true], predicted=SPECIES[pred])
            misclassified.append(row)

    train_frame = X_train.assign(species=y_train.map(dict(enumerate(SPECIES))))
    metrics = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "dataset": {
            "n_samples": int(len(df)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
        },
        "cv_folds": CV_FOLDS,
        "cv_results": cv_results,
        "selected_model": chosen,
        "selection_rule": (
            f"Highest mean {CV_FOLDS}-fold CV accuracy on the training set; "
            f"models within {SELECTION_TOLERANCE} of the best are tie-broken in favour "
            "of the simpler model."
        ),
        "test": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            # 95% Wilson interval: with only 30 test flowers the accuracy estimate is coarse.
            "accuracy_ci95_wilson": accuracy_interval(int((y_test == y_pred).sum()), len(y_test)),
            "precision_macro": float(precision_score(y_test, y_pred, average="macro")),
            "recall_macro": float(recall_score(y_test, y_pred, average="macro")),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(
                y_test, y_pred, target_names=SPECIES, output_dict=True
            ),
            "misclassified": misclassified,
        },
        "permutation_importance": {
            f: {"mean": float(m), "std": float(s)}
            for f, m, s in zip(FEATURES, perm.importances_mean, perm.importances_std)
        },
        "train_stats": {
            "min": {f: float(v) for f, v in X_train.min().items()},
            "max": {f: float(v) for f, v in X_train.max().items()},
            "species_means": {
                sp: {f: float(v) for f, v in grp[FEATURES].mean().items()}
                for sp, grp in train_frame.groupby("species")
            },
        },
    }
    return model, metrics


def main() -> dict:
    model, metrics = train_and_evaluate()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Cross-validation accuracy (training set):")
    for name, r in metrics["cv_results"].items():
        print(f"  {name:<30} {r['mean']:.3f} +/- {r['std']:.3f}")
    print(f"Selected model: {metrics['selected_model']}")
    t = metrics["test"]
    print(f"Test accuracy: {t['accuracy']:.3f} | macro F1: {t['f1_macro']:.3f}")
    print("Confusion matrix (rows = true, cols = predicted):")
    print(np.array(t["confusion_matrix"]))
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")
    return metrics


if __name__ == "__main__":
    main()
