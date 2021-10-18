from typing import Dict, Tuple, Union

import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, fowlkes_mallows_score, \
    rand_score, silhouette_score
from sklearn.neighbors import LocalOutlierFactor


def clusterize(
        dataset: pd.DataFrame, clusterizer: Union[KMeans, AgglomerativeClustering],
        db_scan: DBSCAN, lof: LocalOutlierFactor
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    dataset_labels = dataset.iloc[:, -1]
    return (
        _calculate_statistics(clusterizer, dataset, dataset_labels),
        _calculate_statistics(db_scan, dataset, dataset_labels),
        _calculate_statistics(lof, dataset, dataset_labels)
    )


def _calculate_statistics(
        algorithm: Union[KMeans, AgglomerativeClustering, DBSCAN, LocalOutlierFactor],
        dataset: pd.DataFrame, dataset_labels: pd.Series) -> Dict[str, float]:
    cluster_labels = algorithm.fit_predict(dataset)
    return {
        "silhouette": round(silhouette_score(dataset, cluster_labels), 3),
        "calinski_harabasz": round(calinski_harabasz_score(dataset, cluster_labels), 3),
        "davies_bouldin": round(davies_bouldin_score(dataset, cluster_labels), 3),
        "rand_score": round(rand_score(dataset_labels, cluster_labels), 3),
        "fowlkes_mallows": round(fowlkes_mallows_score(dataset_labels, cluster_labels), 3)
    }
