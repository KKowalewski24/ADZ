from typing import Any, Dict

from pyod.models.abod import ABOD

from module.detector.Detector import Detector


class FastAbodDetector(Detector):

    def detect(self, params: Dict[str, Any]) -> None:
        # Default param for ABOD is method="fast"
        abod = ABOD(**params)
        abod.fit(self.X_train)

        self.y_train_pred = abod.labels_
        self.y_train_scores = abod.decision_scores_

        self.y_test_pred = abod.predict(self.X_test)
        self.y_test_scores = abod.decision_function(self.X_test)
