import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess_data(input_filepath: str, output_filepath: str) -> None:
    # read file
    df = pd.read_csv(input_filepath)

    # encode date
    df["Date"] = pd.to_datetime(df['Date']).map(lambda date: date.month)

    # encode text columns
    label_encoder = LabelEncoder()
    for column_name in ["Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday", "RainTomorrow"]:
        df[column_name] = label_encoder.fit_transform(df[column_name])

    # impute missing numerical values using mean value
    df = df.fillna(df.mean())

    # save processed data to file
    df.to_csv(output_filepath, index=False)
