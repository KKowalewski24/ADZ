import os
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

DATASET_DIR: str = "data/"


# https://www.kaggle.com/rakannimer/air-passengers
def read_air_passengers() -> Tuple[pd.DataFrame, np.ndarray]:
    indexes = [6, 33, 36, 51, 60, 100, 135]
    values = [205, 600, 150, 315, 150, 190, 620]

    return _add_outliers_set_datetime(
        pd.read_csv(f"{DATASET_DIR}air_passengers.csv"), indexes, values, "date", "passengers"
    )


# https://www.kaggle.com/bulentsiyah/for-simple-exercises-time-series-forecasting?select=Alcohol_Sales.csv
def read_alcohol_sales() -> Tuple[pd.DataFrame, np.ndarray]:
    indexes = [
        72, 128, 151, 208, 253, 315
    ]
    values = [
        3000, 2000, 10000, 8300, 12180, 9000,
    ]

    return _add_outliers_set_datetime(
        pd.read_csv(f"{DATASET_DIR}alcohol_sales.csv"), indexes, values, "date", "sales_number"
    )


# https://www.kaggle.com/arashnic/learn-time-series-forecasting-from-gold-price?select=gold_price_data.csv
def read_gold_price() -> Tuple[pd.DataFrame, np.ndarray]:
    path = f"{DATASET_DIR}gold_price_data.csv"
    path_preprocessing = _add_suffix(path)
    _gold_price_preprocessing(path, path_preprocessing)

    indexes = [
        37, 140, 220, 306, 404, 441
    ]
    values = [
        800, 250, 150, 500, 1350, 120
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
