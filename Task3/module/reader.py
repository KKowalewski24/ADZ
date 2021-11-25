import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATASET_DIR: str = "data/"


# https://www.kaggle.com/rakannimer/air-passengers
def read_oil() -> Tuple[pd.DataFrame, np.ndarray]:
    oil_data = [
        111.0091, 130.8284, 141.2871, 154.2278, 162.7409, 192.1665, 240.7997, 304.2174,
        384.0046, 429.6622, 359.3169, 437.2519, 468.4008, 424.4353, 487.9794, 509.8284,
        506.3473, 340.1842, 240.2589, 219.0328, 172.0747, 252.5901, 221.0711, 276.5188,
        271.1480, 342.6186, 428.3558, 442.3946, 432.7851, 437.2497, 437.2092, 445.3641,
        453.1950, 454.4096, 422.3789, 456.0371, 440.3866, 425.1944, 486.2052, 500.4291,
        521.2759, 508.9476, 488.8889, 509.8706, 456.7229, 473.8166, 525.9509, 549.8338,
        542.3405,
    ]

    # Add outliers
    indexes = [4, 6, 7, 25, 40]
    values = [2, 84220, 59598, 46, 12]
    for index, value in zip(indexes, values):
        oil_data[index] = value

    df = pd.DataFrame(oil_data, pd.date_range("1965", "2013", freq="AS"))
    return df, _get_ground_truth_array(df, indexes)


# https://www.kaggle.com/garystafford/environmental-sensor-data-132k
def read_env_telemetry() -> Tuple[pd.DataFrame, np.ndarray]:
    path = f"{DATASET_DIR}iot_telemetry_data.csv"
    path_preprocessing = _add_suffix(path)

    if not os.path.exists(path_preprocessing):
        df = pd.read_csv(path, nrows=2000)
        df.drop(columns=["device"], inplace=True)
        df["ts"] = df["ts"].astype(str)
        _encode_labels(df, ["light", "motion"])
        df.to_csv(path_preprocessing, index=False)

    df = pd.read_csv(path_preprocessing)
    indexes = []

    return df, _get_ground_truth_array(df, indexes)


# https://www.kaggle.com/jsphyg/weather-dataset-rattle-package
def read_weather_aus() -> Tuple[pd.DataFrame, np.ndarray]:
    path = f"{DATASET_DIR}weatherAUS.csv"
    path_preprocessing = _add_suffix(path)

    if not os.path.exists(path_preprocessing):
        df = pd.read_csv(path, nrows=2000)
        _encode_labels(
            df, ["Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday", "RainTomorrow"]
        )
        df = df.fillna(df.mean())
        df.to_csv(path_preprocessing, index=False)

    df = pd.read_csv(path_preprocessing)[["MaxTemp"]]
    indexes = []

    return df, _get_ground_truth_array(df, indexes)


def _get_ground_truth_array(df: pd.DataFrame, indexes: List[int]) -> np.ndarray:
    y = np.zeros(len(df.index))
    np.put(y, indexes, 1)
    return y


def _encode_labels(df: pd.DataFrame, column_names: List[str]) -> None:
    label_encoder = LabelEncoder()
    for column_name in column_names:
        df[column_name] = label_encoder.fit_transform(df[column_name])


def _add_suffix(path: str) -> str:
    return path.replace(".csv", "_preprocessed.csv")
