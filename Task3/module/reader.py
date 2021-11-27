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
        72, 107, 137, 213, 235, 276, 301, 315, 462, 596, 613, 633, 695, 789, 835, 873, 14372, 15134,
        15203, 15334, 16472, 16639, 16782, 19319, 19405, 20669, 20791, 22389, 22639, 22906, 24934,
        25000
    ]

    return _add_outliers_set_datetime(
        pd.read_csv(f"{DATASET_DIR}alcohol_sales.csv"), indexes, values, "date", "sales_number"
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
        10631, 10650
    ]

    values = [
        1, 2, 2, 3, 7, 14, 15, 16, 19, 20, 30, 30, 31, 33, 34, 35, 37, 40, 41, 44, 46, 46, 46, 52, 54,
        55, 56, 57, 61, 61, 65, 66, 67, 68, 69, 76, 84, 87, 88, 89, 89, 94, 96, 96, 99, 102, 102, 103,
        104, 105, 106, 108, 110, 110, 112, 112, 112, 117, 118, 119, 119, 120, 124, 148, 151, 151, 152,
        153, 156, 156, 158, 161, 162, 164, 165, 167, 167, 168, 169, 169, 171, 173, 176, 177, 178, 178,
        179, 180, 181, 182, 182, 185, 186, 187, 188, 188, 189, 191, 197, 200, 1558, 1586, 1696, 1717,
        1788, 1886, 2047, 2174, 2428, 2500, 2677, 2689, 2892, 2947, 3051, 3193, 3346, 3432, 3588,
        3638, 3689, 3757, 4113, 4175, 4236, 4354, 4673, 4691, 4973, 4988, 5068, 5135, 5150, 5242,
        5355, 5373, 5515, 5652, 5893, 5970, 6074, 6146, 6391, 6598, 6657, 6939, 6939, 7224, 7292,
        7390, 7414, 7434, 7708, 7756, 7772, 7774, 7780, 7942, 8064, 8081, 8111, 8155, 8185, 8365,
        8484, 8672, 8942, 9683, 10174, 10200, 10315, 10364, 10628, 10643, 10671, 10672, 10752, 10791,
        10961, 10998, 11055, 11412, 11838, 11854, 11868, 11940, 12024, 12051, 12083, 12161, 12237,
        12260, 12350, 12469, 12471, 12655, 12729, 13062, 13106, 13112, 13129, 13265, 13396, 13603,
        13817, 13883, 13906, 13961, 13977, 13995, 14080, 14141, 14522, 14707, 14982
    ]

    return _add_outliers_set_datetime(
        pd.read_csv(f"{DATASET_DIR}gold_price_data.csv"), indexes, values, "date", "price"
    )


def _add_outliers_set_datetime(
        df: pd.DataFrame, indexes: List[int], values: List[Union[int, float]],
        date_column_name: str, value_column_name: str
) -> Tuple[pd.DataFrame, np.ndarray]:
    if len(indexes) != len(values):
        raise Exception("Arrays must have equals length!")

    df[date_column_name] = pd.to_datetime(df[date_column_name])

    for index, value in zip(indexes, values):
        df.loc[index, value_column_name] = value

    return df, _get_ground_truth_array(df, indexes)


def _get_ground_truth_array(df: pd.DataFrame, indexes: List[int]) -> np.ndarray:
    y = np.zeros(len(df.index))
    np.put(y, indexes, 1)
    return y
