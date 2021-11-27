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
        23156, 19940, 14759, 19987, 19269, 22246, 17283, 20015, 18974, 15473, 16370, 18360, 15439,
        14853, 24657, 24940, 1263, 331, 322, 1669, 1059, 1625, 1432, 2440, 2324, 1650, 1242, 408, 830,
        315, 2068, 303,
    ]

    return _add_outliers_set_datetime(
        pd.read_csv(f"{DATASET_DIR}alcohol_sales.csv"), indexes, values, "date", "sales_number"
    )


# https://www.kaggle.com/arashnic/learn-time-series-forecasting-from-gold-price?select=gold_price_data.csv
def read_gold_price() -> Tuple[pd.DataFrame, np.ndarray]:
    path = f"{DATASET_DIR}gold_price_data.csv"
    path_preprocessing = _add_suffix(path)
    _gold_price_preprocessing(path, path_preprocessing)

    # 10% of outliers
    indexes = [
        5, 14, 21, 26, 29, 31, 34, 51, 58, 71, 73, 94, 153, 156, 172, 182, 198, 209, 209, 211, 212,
        228, 233, 233, 238, 253, 267, 267, 268, 275, 280, 284, 291, 291, 298, 300, 314, 319, 355, 370,
        388, 398, 398, 416, 416, 439, 446, 464, 468, 477, 477, 488, 514
    ]

    values = [
        2228, 4500, 4552, 3386, 2730, 3011, 2881, 3701, 3705, 2272, 2632, 1303, 1660, 1628, 3236,
        2800, 1872, 2483, 1741, 3914, 2463, 4273, 4802, 4187, 1480, 4497, 2951, 4158, 3089, 2442, 21,
        23, 3, 8, 14, 23, 28, 15, 3, 13, 18, 26, 26, 15, 4, 3, 30, 9, 5, 25, 24, 27, 9,
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
