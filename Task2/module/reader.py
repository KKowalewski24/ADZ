import numpy as np
import pandas as pd

DATASET_DIR: str = "data/"


def read_dataset_1() -> pd.DataFrame:
    # TODO ADD IMPL
    return pd.DataFrame({})


def read_dataset_2() -> pd.DataFrame:
    # TODO ADD IMPL
    return pd.DataFrame({})


def read_dataset_3() -> pd.DataFrame:
    # TODO ADD IMPL
    return pd.DataFrame({})


def read_iris_ds() -> np.ndarray:
    return pd.read_csv("data/Iris.csv").iloc[:, 1:5].to_numpy()
