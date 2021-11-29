import os
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

DATASET_DIR: str = "data/"


# https://www.kaggle.com/rakannimer/air-passengers
def read_air_passengers() -> Tuple[pd.DataFrame, np.ndarray]:
    # 11% of outliers
    indexes = [6, 33, 36, 51, 60, 121, 122, 123, 124, 135, 136, 137, 138, 139, 140, 141]
    values = [205, 600, 150, 315, 150, 340, 340, 340, 340, 620, 620, 620, 620, 620, 620, 620]

    return _add_outliers_set_datetime(
        pd.read_csv(f"{DATASET_DIR}air_passengers.csv"), indexes, values, "date", "passengers"
    )


# https://www.kaggle.com/bulentsiyah/for-simple-exercises-time-series-forecasting?select=Alcohol_Sales.csv
def read_alcohol_sales() -> Tuple[pd.DataFrame, np.ndarray]:
    # 6% of outliers
    indexes = [
        72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 128, 151, 315, 316, 317, 318, 319, 320
    ]
    values = [
        5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 2000, 8300, 9000,
        9000, 9000, 9000, 9000, 9000
    ]

    return _add_outliers_set_datetime(
        pd.read_csv(f"{DATASET_DIR}alcohol_sales.csv"), indexes, values, "date", "sales_number"
    )


# https://www.kaggle.com/arashnic/learn-time-series-forecasting-from-gold-price?select=gold_price_data.csv
def read_gold_price() -> Tuple[pd.DataFrame, np.ndarray]:
    path = f"{DATASET_DIR}gold_price_data.csv"
    path_preprocessing = _add_suffix(path)
    _gold_price_preprocessing(path, path_preprocessing)

    # 4% of outliers
    indexes = [
        37, 38, 140, 141, 142, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 306, 441
    ]
    values = [
        800, 800, 250, 250, 250, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 320, 120
    ]

    return _add_outliers_set_datetime(
        pd.read_csv(path_preprocessing), indexes, values, "date", "price"
    )


def _gold_price_preprocessing(path: str, path_preprocessing: str) -> None:
    if not os.path.exists(path_preprocessing):
        temp_file_path = f"{DATASET_DIR}df_temp.csv"
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        df_temp = round(df.groupby([df["date"].dt.year, df["date"].dt.month]).mean(), 2)
        df_temp.to_csv(temp_file_path)
        df_temp = pd.read_csv(temp_file_path)
        os.remove(temp_file_path)

        df_temp["date"] = (
                df_temp.iloc[:, 0].astype(str) + "-" + df_temp.iloc[:, 1].astype(str)
        ).apply(pd.to_datetime, format="%Y-%m-%d", errors="coerce")
        df_temp.drop(df_temp.columns[1], axis=1, inplace=True)
        df_temp.to_csv(path_preprocessing, index=False)


def _add_outliers_set_datetime(
        df: pd.DataFrame, indexes: List[int], values: List[Union[int, float]],
        date_column_name: str, value_column_name: str
) -> Tuple[pd.DataFrame, np.ndarray]:
    if len(indexes) != len(values):
        raise Exception("Arrays must have equal length!")

    df[date_column_name] = pd.to_datetime(df[date_column_name])

    for index, value in zip(indexes, values):
        df.loc[index, value_column_name] = value

    return df, _get_ground_truth_array(df, indexes)


def _get_ground_truth_array(df: pd.DataFrame, indexes: List[int]) -> np.ndarray:
    y = np.zeros(len(df.index))
    np.put(y, indexes, 1)
    return y


def _add_suffix(path: str) -> str:
    return path.replace(".csv", "_preprocessed.csv")
