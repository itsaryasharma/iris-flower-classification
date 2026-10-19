# Model Card: Iris Species Classifier

## Overview
| | |
|---|---|
| Task | Multi-class classification: *Iris setosa*, *versicolor* or *virginica* from 4 measurements (cm) |
| Model | `Pipeline(StandardScaler, LogisticRegression(max_iter=1000))`: multinomial (softmax), L2, C = 1.0, lbfgs |
| Data | Fisher's Iris dataset via `sklearn.datasets.load_iris` (150 flowers, 50 per species, no missing values) |
| Code | `src/train.py` (training and evaluation), `src/predict.py` (validation and inference) |
| Artifacts | `models/iris_model.joblib`, `models/metrics.json` (every number shown in the app) |

## Method
1. Stratified 80/20 split (`random_state=42`): 120 training / 30 test flowers.
2. Four candidates compared with stratified 5-fold cross-validation **on the training set only**.
   Scale-sensitive models include the scaler in their Pipeline, so it is fitted per fold (no leakage).
3. Selection rule (in code): best mean CV accuracy; candidates within 0.01 of the best are tie-broken in favour of the simpler model.
4. The chosen pipeline is refitted on all training data and evaluated **once** on the test set.

| Candidate | 5-fold CV accuracy (mean ± std) |
|---|---|
| **Logistic Regression (selected)** | 0.958 ± 0.026 |
| k-Nearest Neighbours (k = 5) | 0.958 ± 0.026 |
| Decision Tree (max depth 3) | 0.958 ± 0.026 |
| SVM, RBF kernel (calibrated) | 0.967 ± 0.017 |

The SVM scored marginally higher; the difference is smaller than the fold-to-fold spread and within the tie-break tolerance.

## Results (held-out test set, 30 flowers)
| Metric | Value |
|---|---|
| Accuracy | 0.933 (28/30); 95% Wilson interval 0.787–0.982 |
| Macro precision / recall / F1 | 0.933 / 0.933 / 0.933 |
| Per class F1 | setosa 1.00 · versicolor 0.90 · virginica 0.90 |
| Errors | 2, both versicolor ↔ virginica (dataset rows 77 and 134) |
| Permutation importance (accuracy drop) | petal width 0.216 · petal length 0.190 · sepal width 0.052 · sepal length 0.021 |

Reproduce with `python -m src.train`; `pytest` checks that the stored metrics equal those recomputed from the saved model.

## Intended use and limitations
- An educational demonstration of a complete, explainable classification workflow.
- It always answers with one of the three species (no "unknown" class). Inputs outside the training range trigger an extrapolation warning.
- With 150 flowers and a 30-flower test set, performance estimates are coarse (see the interval above).
- Probabilities express the model's confidence; their calibration was not evaluated.
- The species photos in the app are illustrative only; the model uses only the four measurements.
