from typing import Tuple

import numpy as np
from pyod.models.abod import ABOD

from module.detector.Detector import Detector


class AbodDetector(Detector):

    def detect(self, dataset: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
        X_train, X_test, y_train, y_test = dataset

        abod = ABOD()
        abod.fit(X_train)

        y_train_pred = abod.labels_
        y_train_scores = abod.decision_scores_

        y_test_pred = abod.predict(X_test)
        y_test_scores = abod.decision_function(X_test)

        # return y_train_pred, y_test_pred, y_train_scores, y_test_scores
