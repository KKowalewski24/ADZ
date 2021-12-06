from typing import Any, Dict

from module.detector.Detector import Detector
from module.sklearn_extensions_library.kmeans import FuzzyKMeans


class FuzzyKMeansDetector(Detector):

    def detect(self, params: Dict[str, Any]) -> None:
        fuzzy_kmeans = FuzzyKMeans(k=3)
        fuzzy_kmeans.fit(self.X)
        print(fuzzy_kmeans.cluster_centers_)


    def show_results(self, save_results: bool) -> None:
        pass
