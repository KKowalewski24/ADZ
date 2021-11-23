import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATASET_DIR: str = "data/"


# https://www.kaggle.com/rakannimer/air-passengers
def read_air_passengers() -> Tuple[np.ndarray, np.ndarray]:
    path = f"{DATASET_DIR}AirPassengers.csv"
    path_preprocessing = _add_suffix(path)

    if not os.path.exists(path_preprocessing):
        df = pd.read_csv(path)
        # TODO Consider converting year and month to timestamp
        # df["Month"] = datetime.timestamp(df["Month"])
        df.to_csv(path_preprocessing, index=False)

    df = pd.read_csv(path_preprocessing, index_col=["Month"])

    # Add outliers
    indexes = [4, 6, 7, 39, 40, 50, 52, 79, 91, 92, 105, 110, 117, 136, 137]
    values = [
        797, 84220, 59598, 46, 30620, 635801, 1542152, 70546, 5,
        889870, 67738, 8029, 13743, 460130, 95508
    ]
    for index, value in zip(indexes, values):
        df.iloc[index]["Passengers"] = value

    return df.to_numpy(), _get_ground_truth_array(df, indexes)


# https://www.kaggle.com/garystafford/environmental-sensor-data-132k
def read_env_telemetry() -> Tuple[np.ndarray, np.ndarray]:
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

    return df.to_numpy(), _get_ground_truth_array(df, indexes)


# https://www.kaggle.com/jsphyg/weather-dataset-rattle-package
def read_weather_aus() -> Tuple[np.ndarray, np.ndarray]:
    path = f"{DATASET_DIR}weatherAUS.csv"
    path_preprocessing = _add_suffix(path)

    if not os.path.exists(path_preprocessing):
        df = pd.read_csv(path, nrows=2000)
        _encode_labels(
            df, ["Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday", "RainTomorrow"]
        )
        df.to_csv(path_preprocessing, index=False)

    df = pd.read_csv(path_preprocessing)
    indexes = []

    return df.to_numpy(), _get_ground_truth_array(df, indexes)


def _get_ground_truth_array(df: pd.DataFrame, indexes: List[int]) -> np.ndarray:
    y = np.zeros(len(df.index))
    np.put(y, indexes, 1)
    return y


def _encode_labels(df: pd.DataFrame, column_names: List[str]) -> pd.DataFrame:
    label_encoder = LabelEncoder()
    for column_name in column_names:
        df[column_name] = label_encoder.fit_transform(df[column_name])

    return df


def _add_suffix(path: str) -> str:
    return path.replace(".csv", "_preprocessed.csv")
