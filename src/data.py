"""Dataset loading.

The Iris dataset ships with scikit-learn, so no download is needed and the
project works offline. scikit-learn's copy follows Fisher's original 1936
paper (the UCI copy contains two transcription errors).
"""
import pandas as pd
from sklearn.datasets import load_iris

from src.config import FEATURES


def load_iris_frame() -> pd.DataFrame:
    """Return the 150-row Iris dataset as a DataFrame.

    Columns: the four FEATURES, `target` (0/1/2) and `species` (name).
    """
    bunch = load_iris(as_frame=True)
    df = bunch.data.copy()
    df.columns = FEATURES  # e.g. "sepal length (cm)" -> "sepal_length"
    df["target"] = bunch.target.astype(int)
    df["species"] = df["target"].map(dict(enumerate(bunch.target_names)))
    return df
