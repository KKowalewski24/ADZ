from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np


class Detector(ABC):
    RESULTS_DIR = "results/"


    def __init__(self, dataset: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                 configuration_name: str) -> None:
        self.X_train, self.X_test, self.y_train, self.y_test = dataset
        self.configuration_name = configuration_name
        self.statistics: Dict[str, float] = {}


    @abstractmethod
    def detect(self, params: Dict[str, Any]) -> None:
        pass


    @abstractmethod
    def calculate_statistics(self) -> Dict[str, float]:
        pass


    @abstractmethod
    def show_results(self, save_results: bool) -> None:
        pass


    def _statistics_to_string(self) -> str:
        return "_".join([
            stat + '=' + str(self.statistics[stat]).replace('.', ',')
            for stat in self.statistics
        ])
