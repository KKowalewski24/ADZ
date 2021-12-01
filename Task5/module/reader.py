import os
from typing import Tuple

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat

DATASET_DIR: str = "data/"
SYNTHETIC_DATASET_PATH = f"{DATASET_DIR}synthetic_dataset.csv"


def read_synthetic_dataset() -> Tuple[np.ndarray, np.ndarray]:
    # TODO
    # if not os.path.exists(SYNTHETIC_DATASET_PATH):
    #     pass

    # data = pd.read_csv(SYNTHETIC_DATASET_PATH)
    pass


def read_http_dataset() -> Tuple[np.ndarray, np.ndarray]:
    file = h5py.File(f"{DATASET_DIR}http.mat")
    X = np.array(file["X"]).transpose()
    y = np.array(file["y"]).transpose().squeeze().astype(np.int32)

    np.random.seed(47)
    X_normal = _random_samples(X[y == 0], 0.01)
    X_outliers = _random_samples(X[y == 1], 0.02)
    return (
        np.concatenate([X_normal, X_outliers]),
        np.concatenate([np.zeros((len(X_normal),)), np.zeros((len(X_outliers),)) - 1]).astype(
            np.int32)
    )


def read_mammography_dataset() -> Tuple[np.ndarray, np.ndarray]:
    file = loadmat(f"{DATASET_DIR}mammography.mat")
    X = file["X"]
    y = file["y"].squeeze().astype(np.int32)
    y[y == 1] = -1
    return X, y


def _random_samples(X, fraction):
    return X[np.random.randint(len(X), size=(int(fraction * len(X)),))]
