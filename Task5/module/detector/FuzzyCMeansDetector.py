from typing import Any, Dict

import numpy as np
from fcmeans import FCM

from module.detector.Detector import Detector


class FuzzyCMeansDetector(Detector):

    def detect(self, params: Dict[str, Any]) -> None:
        outlier_fraction_threshold = params["outlier_fraction_threshold"]
        fcm = FCM(**params)
        fcm.random_state = 21
        fcm.fit(self.X)
        self.y_pred = fcm.predict(self.X)

        for label in np.unique(self.y_pred):
            quantity = np.count_nonzero(self.y_pred == label)
            if quantity < outlier_fraction_threshold * len(self.X):
                self.y_pred[self.y_pred == label] = -1
