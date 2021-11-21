from typing import Tuple

import numpy as np
from pyod.models.abod import ABOD
from pyod.utils.example import visualize

from module.detector.Detector import Detector


class AbodDetector(Detector):

    def detect(self, dataset: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
        self.X_train, self.X_test, self.y_train, self.y_test = dataset

        abod = ABOD()
        abod.fit(self.X_train)

        self.y_train_pred = abod.labels_
        self.y_train_scores = abod.decision_scores_

        self.y_test_pred = abod.predict(self.X_test)
        self.y_test_scores = abod.decision_function(self.X_test)


    def show_results(self, save_results: bool) -> None:
        visualize(
            type(self).__name__, self.X_train, self.y_train, self.X_test, self.y_test,
            self.y_train_pred, self.y_test_pred, show_figure=not save_results,
            save_figure=save_results
        )
