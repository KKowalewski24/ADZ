import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATASET_DIR: str = "data/"


# https://www.kaggle.com/parulpandey/palmer-archipelago-antarctica-penguin-data?select=penguins_size.csv
def read_penguins_dataset() -> pd.DataFrame:
    df = pd.read_csv(f"{DATASET_DIR}penguins_size.csv")

    label_encoder = LabelEncoder()
    for column_name in ["species", "island", "sex"]:
        df[column_name] = label_encoder.fit_transform(df[column_name])

    return df.fillna(df.mean())


def read_dataset_3() -> pd.DataFrame:
    # TODO ADD IMPL
    return pd.DataFrame({})


def read_synthetic_dataset() -> pd.DataFrame:
    np.random.seed(42)
    X_inliers = 0.3 * np.random.randn(200, 2)
    X_inliers = np.r_[X_inliers + 2, X_inliers - 2]
    X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

    return pd.DataFrame(np.r_[X_inliers, X_outliers])


def read_iris_ds() -> pd.DataFrame:
    return pd.read_csv(f"{DATASET_DIR}Iris.csv").iloc[:, 1:5]
