from typing import List, Tuple

from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.neighbors import LocalOutlierFactor


def resolve_clusterizer_type(clusterizer_names: List[str], chosen_clusterizer_name: str):
    # TODO Consider setting different params
    if clusterizer_names[0] == chosen_clusterizer_name:
        return KMeans()
    elif clusterizer_names[1] == chosen_clusterizer_name:
        return AgglomerativeClustering()


def prepare_benchmark_algorithms() -> Tuple[List, List[str]]:
    # TODO Consider setting different params
    return (
        [
            DBSCAN(),
            LocalOutlierFactor()
        ],
        ["db_scan", "lof"]
    )
