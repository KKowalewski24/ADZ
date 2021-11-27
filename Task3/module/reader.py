import os
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

DATASET_DIR: str = "data/"


# https://www.kaggle.com/rakannimer/air-passengers
def read_air_passengers() -> Tuple[pd.DataFrame, np.ndarray]:
    # 10% of outliers
    indexes = [4, 6, 7, 39, 40, 50, 52, 79, 91, 92, 105, 110, 117, 136, 137]
    values = [
        797, 84220, 59598, 46, 30620, 63580, 154215, 70546, 50,
        88987, 67738, 8029, 13743, 460130, 95508
    ]

    return _add_outliers_set_datetime(
        pd.read_csv(f"{DATASET_DIR}air_passengers.csv"), indexes, values, "date", "passengers"
    )


# https://www.kaggle.com/bulentsiyah/for-simple-exercises-time-series-forecasting?select=Alcohol_Sales.csv
def read_alcohol_sales() -> Tuple[pd.DataFrame, np.ndarray]:
    # 10% of outliers
    indexes = [
        7, 32, 36, 42, 60, 61, 71, 82, 87, 103, 105, 105, 108, 111, 115, 129, 137, 141, 150, 180,
        183, 185, 187, 204, 215, 218, 230, 235, 245, 268, 268, 322
    ]

    values = [

    ]

    return _add_outliers_set_datetime(
        pd.read_csv(f"{DATASET_DIR}alcohol_sales.csv"), indexes, values, "date", "sales_number"
    )


# https://www.kaggle.com/arashnic/learn-time-series-forecasting-from-gold-price?select=gold_price_data.csv
def read_gold_price() -> Tuple[pd.DataFrame, np.ndarray]:
    path = f"{DATASET_DIR}gold_price_data.csv"
    path_preprocessing = _add_suffix(path)

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
        df.to_csv(path_preprocessing, index=False)

    # 10% of outliers
    indexes = [

    ]

    values = [

    ]

    return _add_outliers_set_datetime(
        pd.read_csv(path_preprocessing), indexes, values, "date", "price"
    )


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
