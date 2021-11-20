from typing import Tuple

import numpy as np
import pandas as pd

DATASET_DIR: str = "data/"


# https://www.kaggle.com/rakannimer/air-passengers
def read_air_passengers() -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(f"{DATASET_DIR}AirPassengers.csv", index_col=["Month"])

    # Add outliers
    indexes = [4, 6, 7, 39, 40, 50, 52, 79, 91, 92, 105, 110, 117, 136, 137]
    values = [
        797, 84220, 59598, 46, 30620, 635801, 1542152, 70546, 5,
        889870, 67738, 8029, 13743, 460130, 95508
    ]
    for index, value in zip(indexes, values):
        df.iloc[index]["Passengers"] = value

    y = np.zeros(len(df.index))
    np.put(y, indexes, 1)

    return df.to_numpy(), y


def read_dataset_2() -> Tuple[np.ndarray, np.ndarray]:
    # TODO ADD IMPL
    pass


def read_dataset_3() -> Tuple[np.ndarray, np.ndarray]:
    # TODO ADD IMPL
    pass
