from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from pyod.utils.example import visualize

from module.utils import prepare_filename


class Detector(ABC):
    RESULTS_DIR = "results/"


    def __init__(self) -> None:
        self.X_train: np.ndarray = np.array([])
        self.y_train: np.ndarray = np.array([])
        self.X_test: np.ndarray = np.array([])
        self.y_test: np.ndarray = np.array([])
        self.y_train_pred: np.ndarray = np.array([])
        self.y_train_scores: np.ndarray = np.array([])
        self.y_test_pred: np.ndarray = np.array([])
        self.y_test_scores: np.ndarray = np.array([])


    @abstractmethod
    def detect(self, dataset: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
        pass


    def show_results(self, save_results: bool) -> None:
        visualize(
            prepare_filename(Detector.RESULTS_DIR + type(self).__name__),
            self.X_train, self.y_train, self.X_test, self.y_test, self.y_train_pred,
            self.y_test_pred, show_figure=not save_results, save_figure=save_results
        )
