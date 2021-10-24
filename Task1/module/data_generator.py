import pandas as pd
from sklearn.preprocessing import LabelEncoder
from skmultiflow.data import AGRAWALGenerator


def generate_data(output_filepath: str, rows_number: int) -> pd.DataFrame:
    X, y = AGRAWALGenerator().next_sample(rows_number)
    df: pd.DataFrame = pd.DataFrame(X)
    df["class"] = y
    df.to_csv(output_filepath, index=False)
    return df


def read_preprocess_data() -> pd.DataFrame:
    # TODO
    df = pd.read_csv("data/weatherAUS.csv")
    label_encoder = LabelEncoder()
    for column_name in df.columns:
        df[column_name] = label_encoder.fit_transform(df[column_name])

    df.to_csv("data/filtered_weather.csv", index=False)
    return df
