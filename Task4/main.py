from typing import Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from pyod.models.abod import ABOD
from pyod.models.cof import COF
from pyod.models.lof import LOF
from pyod.utils.data import generate_data
from scipy.io import loadmat
from sklearn.metrics import precision_recall_curve

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


def find_outliers(dataset, detector, **params):
    X, y = dataset
    d = detector(**params).fit(X)
    y_proba = d.predict_proba(X)[:, 1]
    y_proba[np.isnan(y_proba)] = 0
    return precision_recall_curve(y, y_proba)


def plot_results(synthetic, mammography, http):
    plt.plot(synthetic[0], synthetic[1], label="synthetic")
    plt.plot(mammography[0], mammography[1], label="mammography")
    plt.plot(http[0], http[1], label="http")
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.legend()
    plt.show()


# ABOD experiments
synthetic = find_outliers(read_synthetic_dataset(), ABOD, n_neighbors=50)
mammography = find_outliers(read_mammography_dataset(), ABOD, n_neighbors=50)
http = find_outliers(read_http_dataset(), ABOD, n_neighbors=50)
plot_results(synthetic, mammography, http)

# LOF experiments
synthetic = find_outliers(read_synthetic_dataset(), LOF, n_neighbors=50)
mammography = find_outliers(read_mammography_dataset(), LOF, n_neighbors=50)
http = find_outliers(read_http_dataset(), LOF, n_neighbors=50)
plot_results(synthetic, mammography, http)

# COF experiments
synthetic = find_outliers(read_synthetic_dataset(), COF, n_neighbors=50)
mammography = find_outliers(read_mammography_dataset(), COF, n_neighbors=50)
http = find_outliers(read_http_dataset(), COF, n_neighbors=50)
plot_results(synthetic, mammography, http)
