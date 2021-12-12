from typing import Any, Dict, Tuple

import numpy as np
from pyod.models.abod import ABOD
from pyod.utils import precision_n_scores
from pyod.utils.example import visualize
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.utils import check_consistent_length, column_or_1d

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


    def calculate_statistics(self) -> Dict[str, float]:
        roc, precision = self._evaluate_outlier_detection(self.y_test, self.y_test_scores)
        self.statistics = {
            "roc": roc,
            "precision": precision
        }

        return self.statistics


    def _evaluate_outlier_detection(self, y, y_pred) -> Tuple[float, float]:
        y = column_or_1d(y)
        y_pred = column_or_1d(y_pred)
        check_consistent_length(y, y_pred)
        return (
            np.round(roc_auc_score(y, y_pred), decimals=4),
            np.round(precision_n_scores(y, y_pred), decimals=4)
        )


    def show_results(self, save_results: bool) -> None:
        tsne = TSNE(n_components=2)
        filename = prepare_filename(
            Detector.RESULTS_DIR + self.configuration_name + self._statistics_to_string()
        )
        visualize(
            filename,
            tsne.fit_transform(self.X_train), self.y_train,
            tsne.fit_transform(self.X_test), self.y_test,
            self.y_train_pred, self.y_test_pred,
            show_figure=not save_results, save_figure=save_results
        )
