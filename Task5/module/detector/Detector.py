from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import precision_score, recall_score

from module.utils import prepare_filename


class Detector(ABC):
    RESULTS_DIR = "results/"
    RANDOM_STATE_VALUE = 21


    def __init__(self, dataset: Tuple[np.ndarray, np.ndarray], configuration_name: str) -> None:
        self.X, self.y = dataset
        self.y_pred: np.ndarray = np.ndarray([])
        self.configuration_name = configuration_name
        self.statistics: Dict[str, float] = {}


    @abstractmethod
    def detect(self, params: Dict[str, Any]) -> None:
        pass


    def _mark_outliers(self, outlier_fraction_threshold: float) -> None:
        for label in np.unique(self.y_pred):
            quantity = np.count_nonzero(self.y_pred == label)
            if quantity < outlier_fraction_threshold * len(self.X):
                self.y_pred[self.y_pred == label] = -1


    def calculate_statistics(self) -> Dict[str, float]:
        recall = recall_score(self.y, self.y_pred, average=None, zero_division=0)
        precision = precision_score(self.y, self.y_pred, average=None, zero_division=0)

        # [0] is taken because stats are calculated for (-1) class
        self.statistics = {
            "recall": np.round(recall[0], 2),
            "precision": np.round(precision[0], 2)
        }

        return self.statistics


    def show_results(self, save_results: bool, size: int = 20) -> None:
        # Reduce dimensionality only when there more than 2 dimensions
        if self.X[0].size > 2:
            self.X = TSNE().fit_transform(self.X)

        plt.figure(figsize=(10, 6))
        plt.scatter(self.X[self.y_pred == -1, 0], self.X[self.y_pred == -1, 1], c="k", s=size)
        # All labels except (-1)
        for label in set(np.unique(self.y_pred)) - {-1}:
            plt.scatter(self.X[self.y_pred == label, 0], self.X[self.y_pred == label, 1], s=size)

        self._set_descriptions(self.configuration_name + self._statistics_to_string())
        self._show_and_save(self.configuration_name, Detector.RESULTS_DIR, save_results)


    def _statistics_to_string(self) -> str:
        return "_".join([
            stat + '=' + str(self.statistics[stat]).replace('.', ',')
            for stat in self.statistics
        ])


    def _set_descriptions(self, title: str, x_label: str = "", y_label: str = "") -> None:
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)


    def _show_and_save(self, name: str, results_dir: str, save_data: bool) -> None:
        if save_data:
            plt.savefig(results_dir + prepare_filename(name))
            plt.close()
        plt.show()
