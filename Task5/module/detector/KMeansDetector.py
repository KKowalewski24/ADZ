from typing import Any, Dict

import numpy as np
from sklearn.cluster import KMeans

from module.detector.Detector import Detector


class KMeansDetector(Detector):

    def detect(self, params: Dict[str, Any]) -> None:
        outlier_fraction_threshold = params["outlier_fraction_threshold"]
        params.pop("outlier_fraction_threshold")

        k_means: KMeans = KMeans(**params)
        k_means.random_state = Detector.RANDOM_STATE_VALUE
        self.y_pred = k_means.fit_predict(self.X).astype(np.float32)
        self._mark_outliers(outlier_fraction_threshold)
