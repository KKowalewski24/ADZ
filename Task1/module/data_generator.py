import pandas as pd
from skmultiflow.data import AGRAWALGenerator


def generate_data(output_filepath: str, rows_number: int) -> pd.DataFrame:
    X, y = AGRAWALGenerator().next_sample(rows_number)
    df: pd.DataFrame = pd.DataFrame(X)
    df["class"] = y
    df.to_csv(output_filepath)
    return df
