# Development Log: Iris Species Classifier

A chronological record of what was actually done, why, and what happened. All work took place on
**19 October 2026**.

## Environment
- Windows 11, Python 3.13.1.
- Dependencies installed in a fresh virtual environment: Streamlit 1.66.0, scikit-learn 1.9.1,
  pandas 3.0.6, NumPy 2.5.4, Plotly 7.1.0, joblib 1.6.0, pytest 9.1.1. These are pinned in `requirements.txt`.

## 1. Planning decisions
- **Framework: Streamlit.** It gives an interactive web UI in pure Python. Flask/React was rejected because
  of the one-day deadline; Gradio was rejected because it is less suited to multi-tab dashboards.
- **Data source: `sklearn.datasets.load_iris`.** No download is needed and it works offline. Checked the
  dataset description: scikit-learn's copy follows Fisher's paper, unlike the UCI copy, which has two
  wrong points.
- **Consistent app layout** (shared with the other internship projects): banner → tabs *Predict / Model
  card / How it works / Dataset*.
- **Honesty mechanism:** the training script writes every reported number to `models/metrics.json`, and
  the app reads only from there.

## 2. Implementation
1. `src/config.py`: constants (paths, feature names, seed 42, test size 0.2, 5 CV folds, 0–30 cm validation limits).
2. `src/data.py`: loader that renames columns to snake_case.
3. `src/train.py`: stratified 80/20 split, 5-fold stratified CV of 4 candidates (logistic regression,
   k-NN, decision tree, SVM), a selection rule, refit, a single test evaluation, permutation importance,
   and saving of artifacts.
4. `src/predict.py`: validation, out-of-range warnings, prediction, decision-region grid.
5. `app.py` + `src/ui.py`: the Streamlit interface.
6. Tests: `tests/test_model.py` and `tests/test_app.py`.

## 3. First training run (system Python, scikit-learn 1.6.1, smoke test only)
CV accuracy: LR 0.958, k-NN 0.958, DT 0.958, SVM 0.967. Selected: Logistic Regression (within the 0.01
tolerance). Test accuracy 0.933, with 2 errors (versicolor ↔ virginica).

**Decision:** keep the tie-break rule. The SVM's lead (0.009) is smaller than the fold standard deviation (0.026).

## 4. Problems encountered and fixes

| # | Problem | Cause | Fix |
|---|---|---|---|
| 1 | The first `app.py` draft called `st.stop()` inside the Predict tab on invalid input | That would have stopped the other three tabs from rendering | Restructured the app into `render_*` functions; invalid input now shows an error and skips only the result section |
| 2 | `FutureWarning: The probability parameter was deprecated in 1.9` when training in the pinned environment | scikit-learn 1.9 deprecated `SVC(probability=True)` | Replaced it with `CalibratedClassifierCV(SVC(kernel="rbf"), ensemble=False)`, the officially recommended replacement. Re-ran training with `python -W error`: no warnings, identical results |
| 3 | Uncertain whether Streamlit 1.66 accepts `width="stretch"` | API change in recent Streamlit versions (`use_container_width` is being replaced) | Confirmed by running the app headlessly with `AppTest` and a real server: no errors or deprecation warnings |

## 5. Final training run (pinned environment)
```
Logistic Regression            0.958 +/- 0.026
k-Nearest Neighbours (k=5)     0.958 +/- 0.026
Decision Tree (max depth 3)    0.958 +/- 0.026
SVM (RBF kernel)               0.967 +/- 0.017
Selected model: Logistic Regression
Test accuracy: 0.933 | macro F1: 0.933
```
The saved model file is 1.9 KB and `metrics.json` is 4.5 KB, both committed.

## 6. Verification
- `pytest`: **18 passed** (see [TESTING.md](TESTING.md)).
- Started `streamlit run app.py` headlessly: health endpoint returned `ok` and the page returned HTTP 200.
- Manually verified the worked softmax example in the technical guide against `model.predict_proba`
  (0.9808 / 0.0192 / 0.0000).
- Checked dataset facts quoted in the docs: the duplicate rows (101 and 142) and the borderline example (row 83).
- Installed `requirements.txt` into a brand-new virtual environment and ran the full suite there: **18 passed**.
  Two environment problems occurred on the way, neither caused by the project: a transient PyPI error
  during parallel installs (a retry succeeded), and `WinError 206: filename too long` when the environment
  was placed at a very long path. The second one led to the "Windows note" in the README.

## 7. Not done / possible future work
- No "unknown species" detection (documented limitation).
- No batch CSV prediction (deliberately dropped to protect the deadline).
- The app has not been tested in a mobile browser.
- Screenshots for the report still need to be captured manually (see the list in TESTING.md).

## 8. Refinement phase (19 October 2026)
- **Independent re-verification:** a confusion matrix recomputed by hand from the saved model matches the stored one; accuracy, macro precision and macro F1 match to 6 decimals. The scaler's mean equals the training-rows mean (no preprocessing leakage). Probabilities sum to 1 for all 150 flowers. Edge inputs (0.1 cm and 10 cm sliders) give warnings; 0 cm and 31 cm are rejected.
- **Accuracy uncertainty:** `train.py` now stores a 95% Wilson interval (78.7%–98.2%), shown on the Model card. SciPy was pinned because it is now imported directly. Retraining changed only this new field.
- **Light/dark mode:** both themes are defined in `.streamlit/config.toml` (switch: ⋮ → Settings → Theme). Charts use transparent backgrounds and colours readable on both; custom cards use translucent styling. Checked with headless-browser screenshots in both modes.
- **Visuals:** illustrative species photos (Wikimedia Commons, CC BY-SA; licences checked via the Commons API; `assets/ATTRIBUTIONS.md`), a probability caption, and readable chart labels.
- **Docs:** added `docs/MODEL_CARD.md`; the long-form study notes are no longer part of the repository; a Windows PowerShell quick start was added to the README.
