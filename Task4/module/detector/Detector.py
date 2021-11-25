from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
from pyod.utils.example import visualize
from sklearn.decomposition import PCA

from module.utils import prepare_filename


class Detector(ABC):
    RESULTS_DIR = "results/"


    def __init__(self, dataset: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                 configuration_name: str) -> None:
        self.X_train, self.X_test, self.y_train, self.y_test = dataset
        self.y_train_pred: np.ndarray = np.array([])
        self.y_train_scores: np.ndarray = np.array([])
        self.y_test_pred: np.ndarray = np.array([])
        self.y_test_scores: np.ndarray = np.array([])
        self.configuration_name = configuration_name


    @abstractmethod
    def detect(self, params: Dict[str, Any]) -> None:
        pass


    def show_results(self, save_results: bool) -> None:
        pca = PCA(n_components=2)
        visualize(
            prepare_filename(Detector.RESULTS_DIR + self.configuration_name),
            pca.fit_transform(self.X_train), self.y_train,
            pca.fit_transform(self.X_test), self.y_test,
            self.y_train_pred, self.y_test_pred,
            show_figure=not save_results, save_figure=save_results
        )
