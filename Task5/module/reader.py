from typing import Tuple

import h5py
import numpy as np
from pyod.utils.data import generate_data
from scipy.io import loadmat

DATASET_DIR: str = "data/"
RANDOM_STATE_VALUE = 21


def read_synthetic_dataset() -> Tuple[np.ndarray, np.ndarray]:
    return generate_data(
        n_train=200,
        n_features=2,
        contamination=0.1,
        train_only=True,
        random_state=RANDOM_STATE_VALUE,
        behaviour="new",
    )


def read_mammography_dataset() -> Tuple[np.ndarray, np.ndarray]:
    file = loadmat(f"{DATASET_DIR}mammography.mat")
    X = file["X"]
    y = file["y"].squeeze().astype(np.int32)
    return X, y


def read_http_dataset() -> Tuple[np.ndarray, np.ndarray]:
    file = h5py.File(f"{DATASET_DIR}http.mat")
    X = np.array(file["X"]).transpose()
    y = np.array(file["y"]).transpose().squeeze().astype(np.int32)

    np.random.seed(47)
    X_normal = _random_samples(X[y == 0], 0.01)
    X_outliers = _random_samples(X[y == 1], 0.02)

    X = np.concatenate([X_normal, X_outliers])
    y = np.concatenate(
        [np.zeros((len(X_normal),)), np.zeros((len(X_outliers),)) + 1]
    ).astype(np.int32)

    return X, y


def _random_samples(X, fraction):
    return X[np.random.randint(len(X), size=(int(fraction * len(X)),))]
