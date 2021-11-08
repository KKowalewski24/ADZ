import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATASET_DIR: str = "data/"


# https://www.kaggle.com/parulpandey/palmer-archipelago-antarctica-penguin-data?select=penguins_size.csv
def read_dataset_penguins() -> pd.DataFrame:
    df = pd.read_csv(f"{DATASET_DIR}penguins_size.csv")

    label_encoder = LabelEncoder()
    for column_name in ["species", "island", "sex"]:
        df[column_name] = label_encoder.fit_transform(df[column_name])

    return df.fillna(df.mean())


def read_dataset_2() -> pd.DataFrame:
    # TODO ADD IMPL
    return pd.DataFrame({})


def read_dataset_3() -> pd.DataFrame:
    # TODO ADD IMPL
    return pd.DataFrame({})


def read_iris_ds() -> pd.DataFrame:
    return pd.read_csv(f"{DATASET_DIR}Iris.csv").iloc[:, 1:5]
