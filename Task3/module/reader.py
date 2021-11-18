import numpy as np
import pandas as pd

DATASET_DIR: str = "data/"


# https://www.kaggle.com/rakannimer/air-passengers
def read_air_passengers() -> np.ndarray:
    return pd.read_csv(
        f"{DATASET_DIR}AirPassengers.csv", parse_dates=["Month"], index_col=["Month"]
    ).to_numpy()


def read_dataset_2() -> np.ndarray:
    # TODO ADD IMPL
    return pd.DataFrame({}).to_numpy()


def read_dataset_3() -> np.ndarray:
    # TODO ADD IMPL
    return pd.DataFrame({}).to_numpy()
