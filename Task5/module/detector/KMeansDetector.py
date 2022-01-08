from typing import Any, Dict

import numpy as np
from sklearn.cluster import KMeans

from module.detector.Detector import Detector


class KMeansDetector(Detector):

    def detect(self, params: Dict[str, Any]) -> None:
        filtered_params = {
            key: value
            for key, value in params.items()
            if key != "outlier_fraction_threshold"
        }

        k_means: KMeans = KMeans(**filtered_params)
        k_means.random_state = Detector.RANDOM_STATE_VALUE
        self.y_pred = k_means.fit_predict(self.X).astype(np.float32)
        self._mark_outliers(params["outlier_fraction_threshold"])
