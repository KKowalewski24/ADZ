from typing import List, Tuple, Union

from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.neighbors import LocalOutlierFactor


def resolve_clusterizer_type(clusterizer_names: List[str],
                             chosen_clusterizer_name: str) -> Union[KMeans, AgglomerativeClustering]:
    # TODO Consider setting different params
    if clusterizer_names[0] == chosen_clusterizer_name:
        return KMeans()
    elif clusterizer_names[1] == chosen_clusterizer_name:
        return AgglomerativeClustering()


def prepare_benchmark_algorithms() -> Tuple[DBSCAN, LocalOutlierFactor]:
    # TODO Consider setting different params
    return (
        DBSCAN(),
        LocalOutlierFactor()
    )
