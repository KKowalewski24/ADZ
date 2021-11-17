import os
from typing import Tuple

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat

SYNTHETIC_DATASET_PATH = "data/synthetic_dataset.csv"


def _random_samples(X, fraction):
    return X[np.random.randint(len(X), size=(int(fraction * len(X)),))]


def read_synthetic_dataset() -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(SYNTHETIC_DATASET_PATH):
        np.random.seed(42)
        X_inliers = 0.3 * np.random.randn(200, 2)
        X_inliers = np.r_[X_inliers + 2, X_inliers - 2]
        X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
        pd.DataFrame(np.r_[X_inliers, X_outliers]).to_csv(SYNTHETIC_DATASET_PATH, index=False)

    data = pd.read_csv(SYNTHETIC_DATASET_PATH)
    return np.array(data), np.concatenate([np.zeros((400,)), np.zeros((20,)) - 1]).astype(np.int32)


def read_http_dataset() -> Tuple[np.ndarray, np.ndarray]:
    file = h5py.File("data/http.mat")
    X = np.array(file["X"]).transpose()
    y = np.array(file["y"]).transpose().squeeze().astype(np.int32)

    np.random.seed(47)
    X_normal = _random_samples(X[y == 0], 0.01)
    X_outliers = _random_samples(X[y == 1], 0.02)
    return (
        np.concatenate([X_normal, X_outliers]),
        np.concatenate([np.zeros((len(X_normal),)), np.zeros((len(X_outliers),)) - 1]).astype(np.int32)
    )


def read_mammography_dataset() -> Tuple[np.ndarray, np.ndarray]:
    file = loadmat("data/mammography.mat")
    X = file["X"]
    y = file["y"].squeeze().astype(np.int32)
    y[y == 1] = -1
    return X, y
