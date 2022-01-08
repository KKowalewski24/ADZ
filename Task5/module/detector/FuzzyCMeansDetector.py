from typing import Any, Dict

from fcmeans import FCM

from module.detector.Detector import Detector


class FuzzyCMeansDetector(Detector):

    def detect(self, params: Dict[str, Any]) -> None:
        fcm = FCM(**params)
        fcm.random_state = Detector.RANDOM_STATE_VALUE
        fcm.fit(self.X)
        self.y_pred = fcm.predict(self.X)
        self._mark_outliers(params["outlier_fraction_threshold"])
