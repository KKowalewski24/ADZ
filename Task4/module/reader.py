from typing import Callable, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_DIR: str = "data/"
TEST_DATA_PERCENTAGE = 0.3
RANDOM_STATE_VALUE = 21


def read_dataset_1() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # TODO
    return _read_and_split("", lambda df: (df.iloc[:, :-1], df.iloc[:, -1]))


def read_dataset_2() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # TODO
    return _read_and_split("", lambda df: (df.iloc[:, :-1], df.iloc[:, -1]))


def read_dataset_3() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # TODO
    return _read_and_split("", lambda df: (df.iloc[:, :-1], df.iloc[:, -1]))


def _read_and_split(
        filename: str, callback: Callable[[pd.DataFrame], Tuple[np.ndarray, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(DATASET_DIR + filename)
    X, y = callback(df)
    return train_test_split(X, y, test_size=TEST_DATA_PERCENTAGE, random_state=RANDOM_STATE_VALUE)
