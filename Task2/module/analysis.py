import pandas as pd


def clusterize(dataset: pd.DataFrame, clusterizer) -> float:
    clusterizer.fit_predict(dataset)
    X_scores = clusterizer.negative_outlier_factor_
    radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
    return radius
