from typing import Any, Dict

from sklearn.cluster import KMeans

from module.detector.Detector import Detector


class KMeansDetector(Detector):

    def detect(self, params: Dict[str, Any]) -> None:
        k_means = KMeans(**params)
