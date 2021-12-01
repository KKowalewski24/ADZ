from typing import Any, Dict, Tuple

import numpy as np
from pyod.models.abod import ABOD
from pyod.utils.example import visualize
from sklearn.decomposition import PCA

from module.detector.Detector import Detector
from module.utils import prepare_filename


class FastAbodDetector(Detector):

    def __init__(self, dataset: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                 configuration_name: str) -> None:
        super().__init__(dataset, configuration_name)
        self.y_train_pred: np.ndarray = np.array([])
        self.y_train_scores: np.ndarray = np.array([])
        self.y_test_pred: np.ndarray = np.array([])
        self.y_test_scores: np.ndarray = np.array([])


    def detect(self, params: Dict[str, Any]) -> None:
        # Default param for ABOD is method="fast"
        abod = ABOD(**params)
        abod.fit(self.X_train)

        self.y_train_pred = abod.labels_
        self.y_train_scores = abod.decision_scores_

        self.y_test_pred = abod.predict(self.X_test)
        self.y_test_scores = abod.decision_function(self.X_test)


    def show_results(self, save_results: bool) -> None:
        pca = PCA(n_components=2)
        # TODO CHANGE FOR SOMETHING LESS CUSTOM
        visualize(
            prepare_filename(Detector.RESULTS_DIR + self.configuration_name),
            pca.fit_transform(self.X_train), self.y_train,
            pca.fit_transform(self.X_test), self.y_test,
            self.y_train_pred, self.y_test_pred,
            show_figure=not save_results, save_figure=save_results
        )
