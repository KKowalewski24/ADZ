from typing import List, Tuple, Union

import numpy as np
import pandas as pd

DATASET_DIR: str = "data/"


# https://www.kaggle.com/rakannimer/air-passengers
def read_air_passengers() -> Tuple[pd.DataFrame, np.ndarray]:
    # 10% of outliers
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
    # 10% of outliers
    indexes = [
        7, 32, 36, 42, 60, 61, 71, 82, 87, 103, 105, 105, 108, 111, 115, 129, 137, 141, 150, 180,
        183, 185, 187, 204, 215, 218, 230, 235, 245, 268, 268, 322
    ]

    values = []

    return _add_outliers(
        pd.read_csv(f"{DATASET_DIR}alcohol_sales.csv"), indexes, values, "sales_number"
    )


# https://www.kaggle.com/arashnic/learn-time-series-forecasting-from-gold-price?select=gold_price_data.csv
def read_gold_price() -> Tuple[pd.DataFrame, np.ndarray]:
    # 2% of outliers
    indexes = [
        19, 19, 63, 138, 332, 385, 411, 587, 614, 717, 864, 873, 918, 932, 1056, 1073, 1084, 1116,
        1125, 1158, 1200, 1226, 1262, 1264, 1322, 1415, 1480, 1485, 1497, 1541, 1542, 1575, 1602,
        1810, 1812, 1827, 19, 74, 2022, 2030, 2057, 2058, 2065, 2132, 2180, 2262, 2345, 2370, 2374,
        2388, 2417, 2528, 2681, 2685, 2932, 2956, 2978, 3097, 3107, 3119, 3128, 3163, 3286, 3331,
        3353, 3384, 3414, 3481, 3586, 3601, 3635, 3635, 3703, 3765, 3856, 3901, 3953, 3966, 4057,
        4221, 4284, 4289, 4305, 4398, 4453, 4488, 4508, 4546, 4555, 4562, 4566, 4569, 4630, 4673,
        4706, 4710, 4755, 4779, 4816, 4857, 4874, 4876, 4946, 4992, 5035, 5055, 5096, 5147, 5207,
        5246, 5381, 5405, 5495, 5519, 5544, 5563, 5567, 5577, 5610, 5675, 5738, 5773, 5796, 5827,
        6078, 6148, 6214, 6252, 6345, 6415, 6499, 6504, 6554, 6565, 6573, 6609, 6731, 6842, 68, 52,
        6954, 6962, 6979, 7034, 7062, 7093, 7099, 7115, 7137, 7166, 7222, 7246, 7311, 7387, 7455,
        7522, 7557, 7623, 7661, 7689, 7715, 7795, 7826, 7890, 7920, 8106, 8300, 8353, 8367, 8450,
        8647, 8691, 8698, 8783, 8789, 8857, 8882, 8895, 8906, 8935, 8989, 9026, 9165, 9355, 9365,
        9419, 9441, 9456, 9456, 9473, 9480, 9500, 9591, 9618, 9697, 9736, 9759, 9805, 9923, 9948,
        9963, 9993, 9995, 10167, 10270, 10337, 10373, 10398, 10406, 10411, 10532, 10617, 10618,
        10631, 10650, 10670, 10785
    ]

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
