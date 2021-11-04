import pandas as pd


def clusterize(dataset: pd.DataFrame, clusterizer) -> None:
    dataset_labels = dataset.iloc[:, -1]
    cluster_labels = clusterizer.fit_predict(dataset)
    pass
