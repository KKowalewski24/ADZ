import pandas as pd


def clusterize(dataset: pd.DataFrame, clusterizer) -> None:
    clusterizer.fit_predict(dataset)
    pass
