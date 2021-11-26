from typing import List, Tuple, Union

import numpy as np
import pandas as pd

DATASET_DIR: str = "data/"


# https://www.kaggle.com/rakannimer/air-passengers
def read_air_passengers() -> Tuple[pd.DataFrame, np.ndarray]:
    indexes = [4, 6, 7, 39, 40, 50, 52, 79, 91, 92, 105, 110, 117, 136, 137]
    values = [
        797, 84220, 59598, 46, 30620, 635801, 1542152, 70546, 5,
        889870, 67738, 8029, 13743, 460130, 95508
    ]

    return _add_outliers(
        pd.read_csv(f"{DATASET_DIR}air_passengers.csv"), indexes, values, "passengers"
    )


# https://www.kaggle.com/bulentsiyah/for-simple-exercises-time-series-forecasting?select=Alcohol_Sales.csv
def read_alcohol_sales() -> Tuple[pd.DataFrame, np.ndarray]:
    indexes = []
    values = []

    return _add_outliers(
        pd.read_csv(f"{DATASET_DIR}alcohol_sales.csv"), indexes, values, "sales_number"
    )


# https://www.kaggle.com/arashnic/learn-time-series-forecasting-from-gold-price?select=gold_price_data.csv
def read_gold_price() -> Tuple[pd.DataFrame, np.ndarray]:
    indexes = []
    values = []

    return _add_outliers(
        pd.read_csv(f"{DATASET_DIR}gold_price_data.csv"), indexes, values, "price"
    )


def _add_outliers(df: pd.DataFrame, indexes: List[int], values: List[Union[int, float]],
                  column_name: str) -> Tuple[pd.DataFrame, np.ndarray]:
    for index, value in zip(indexes, values):
        df.iloc[index][column_name] = value

    return df, _get_ground_truth_array(df, indexes)


def _get_ground_truth_array(df: pd.DataFrame, indexes: List[int]) -> np.ndarray:
    y = np.zeros(len(df.index))
    np.put(y, indexes, 1)
    return y
