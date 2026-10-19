"""Central configuration: paths, feature names and fixed settings.

Keeping these in one place means the training script, the prediction code,
the app and the tests all agree on the same names and file locations.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "iris_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

# Column names used throughout the project (snake_case versions of the
# scikit-learn names) and human-friendly labels for the UI.
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
FEATURE_LABELS = {
    "sepal_length": "Sepal length (cm)",
    "sepal_width": "Sepal width (cm)",
    "petal_length": "Petal length (cm)",
    "petal_width": "Petal width (cm)",
}
SPECIES = ["setosa", "versicolor", "virginica"]
SPECIES_COLORS = {"setosa": "#F59E0B", "versicolor": "#10B981", "virginica": "#7C3AED"}

# Hard validation limits. Real iris flowers are a few centimetres in size;
# anything outside (0, 30] cm is treated as a data-entry error, not a flower.
MIN_CM = 0.0
MAX_CM = 30.0

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
