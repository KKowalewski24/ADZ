import os

import numpy as np
import pandas as pd

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
