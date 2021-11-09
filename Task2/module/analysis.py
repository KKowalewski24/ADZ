import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.neighbors import LocalOutlierFactor


def clusterize(dataset: pd.DataFrame, clusterizer) -> float:
    y_pred = clusterizer.fit_predict(dataset)

    if isinstance(clusterizer, KMeans):
        return -1.0
    if isinstance(clusterizer, AgglomerativeClustering):
        return -1.0
    if isinstance(clusterizer, DBSCAN):
        return -1.0
    if isinstance(clusterizer, LocalOutlierFactor):
        X_scores = clusterizer.negative_outlier_factor_
        # Calculate radius
        return (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
