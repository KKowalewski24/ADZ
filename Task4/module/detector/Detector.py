from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class Detector(ABC):

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


    @abstractmethod
    def show_results(self, save_results: bool) -> None:
        pass
