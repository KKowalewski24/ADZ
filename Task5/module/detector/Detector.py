from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
from matplotlib import pyplot as plt

from module.utils import prepare_filename


class Detector(ABC):
    RESULTS_DIR = "results/"


    def __init__(self, dataset: Tuple[np.ndarray, np.ndarray], configuration_name: str) -> None:
        self.X, self.y = dataset
        self.configuration_name = configuration_name
        self.statistics: Dict[str, float] = {}


    @abstractmethod
    def detect(self, params: Dict[str, Any]) -> None:
        pass


    def calculate_statistics(self) -> Dict[str, float]:
        recall = 0
        precision = 0

        self.statistics = {
            "recall": recall,
            "precision": precision
        }

        return self.statistics


    def show_results(self, save_results: bool) -> None:
        plt.figure(figsize=(10, 6))

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
