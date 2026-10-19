# Testing: Iris Species Classifier

## How to run
```bash
pytest            # all tests
pytest -v         # verbose, one line per test
```
`pytest.ini` puts the project root on the import path, so run it from the project folder.

## Strategy
| Layer | What is tested | File |
|---|---|---|
| Data | Shape (150×6), column names, no missing values, 50 per species | `tests/test_model.py` |
| Model honesty | Accuracy recomputed from the saved model **equals** the value stored in `metrics.json`; the confusion matrix sums to the test-set size | `tests/test_model.py` |
| Model quality | Test accuracy ≥ 0.90 (a regression guard, not a target) | `tests/test_model.py` |
| Prediction | Three known flowers predicted correctly; probabilities sum to 1 | `tests/test_model.py` |
| Validation | Negative, zero, absurd (99 cm), non-numeric, NaN and missing values are all rejected with `ValueError` | `tests/test_model.py` |
| Extrapolation | In-range input gives no warning; petal length 9.5 cm gives exactly one warning | `tests/test_model.py` |
| Selection rule | Near-tie picks the simpler model; a clear winner is picked otherwise | `tests/test_model.py` |
| Decision map | Grid has the right shape and only valid class indices | `tests/test_model.py` |
| **App (headless)** | `app.py` runs without exceptions; 4 tabs render; preset buttons change the prediction (setosa → virginica); an out-of-range slider value shows a warning | `tests/test_app.py` |

The app tests use `streamlit.testing.v1.AppTest`, which executes `app.py` exactly as Streamlit would,
without a browser.

## Results (19 October 2026, Python 3.13.1, pinned requirements)
```
..................                                                       [100%]
18 passed in 12.50s
```

Additional manual checks:
- `python -W error -m src.train` completes with no warnings.
- `streamlit run app.py --server.headless true`: `/_stcore/health` returned `ok` and `/` returned HTTP 200.

## Not covered by automated tests
- Visual appearance and chart rendering in a real browser. Check this manually.
- Behaviour on mobile screen sizes.

## Manual demo checklist and screenshots for the report
1. Predict tab with **Typical setosa** loaded (result card + probability bars).
2. **Borderline case** showing the "Uncertain" note.
3. Decision map with the star marker.
4. Petal length set to 9.5 cm, showing the extrapolation warning.
5. Model card: metrics row + confusion matrix + CV comparison.
6. Dataset tab: scatter matrix.

## Refinement-phase verification (19 October 2026)
Command: `python -m pytest -q -W error::DeprecationWarning -W error::FutureWarning` → **18 passed**.

Additional checks (scripted, results recorded in the development log):
- Confusion matrix and macro metrics recomputed by hand from the saved model's predictions on the documented split: identical to `metrics.json`.
- The scaler's fitted mean equals the training-rows mean, not the full-data mean (no leakage).
- Probabilities sum to 1 for all 150 flowers; edge inputs give warnings or are rejected as designed.
- Both light and dark themes checked with headless-browser screenshots (Predict, Model card, Dataset tabs).

**Fresh-clone check:** only the files Git would publish were copied, a new virtual environment was created with
`py -3.13 -m venv`, `pip install -r requirements.txt` succeeded, `python -m pytest` gave 18 passed, and
`python -m streamlit run app.py` started (health `ok`, HTTP 200).
