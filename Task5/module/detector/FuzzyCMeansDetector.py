from typing import Any, Dict

import numpy as np
from fcmeans import FCM

from module.detector.Detector import Detector


class FuzzyCMeansDetector(Detector):

    def detect(self, params: Dict[str, Any]) -> None:
        fcm = FCM(**params)
        fcm.random_state = Detector.RANDOM_STATE_VALUE
        fcm.fit(self.X)
        self.y_pred = fcm.predict(self.X)
        self._mark_outliers(params["outlier_fraction_threshold"])


    def _mark_outliers(self, outlier_fraction_threshold: float) -> None:
        for label in np.unique(self.y_pred):
            quantity = np.count_nonzero(self.y_pred == label)
            if quantity < outlier_fraction_threshold * len(self.X):
                self.y_pred[self.y_pred == label] = -1
