import os
from typing import List

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

DATASET_DIR: str = "data/"
SYNTHETIC_DATASET_PATH = f"{DATASET_DIR}synthetic_dataset.csv"


def read_synthetic_dataset() -> pd.DataFrame:
    if not os.path.exists(SYNTHETIC_DATASET_PATH):
        np.random.seed(42)
        X_inliers = 0.3 * np.random.randn(200, 2)
        X_inliers = np.r_[X_inliers + 2, X_inliers - 2]
        X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
        pd.DataFrame(np.r_[X_inliers, X_outliers]).to_csv(SYNTHETIC_DATASET_PATH, index=False)

    return pd.read_csv(SYNTHETIC_DATASET_PATH)


def _encode_labels(df: pd.DataFrame, column_names: List[str]) -> pd.DataFrame:
    label_encoder = LabelEncoder()
    for column_name in column_names:
        df[column_name] = label_encoder.fit_transform(df[column_name])

    return df


def _normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    return preprocessing.normalize(df, axis=0)
