from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from module.utils import prepare_filename


class Detector(ABC):

    def __init__(self, dataset: pd.DataFrame, ground_truth_outliers: np.ndarray,
                 configuration_name: str) -> None:
        self.dataset = dataset
        self.ground_truth_outliers = ground_truth_outliers
        self.configuration_name = configuration_name
        self.statistics: Dict[str, float] = {}


    @abstractmethod
    def detect(self, params: Dict[str, Any]) -> None:
        pass


    def _fill_outliers_array(self, dataset_size: int, indexes: List[int]) -> np.ndarray:
        outliers_array = np.zeros(dataset_size)
        np.put(outliers_array, indexes, 1)
        return outliers_array


    @abstractmethod
    def calculate_statistics(self) -> Dict[str, float]:
        pass


    def _statistics_to_string(self) -> str:
        return " ".join([stat + '=' + str(self.statistics[stat]) for stat in self.statistics])


    @abstractmethod
    def show_results(self, results_dir: str, save_data: bool) -> None:
        pass


    def _set_descriptions(self, title: str, x_label: str = "", y_label: str = "") -> None:
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)


    def _show_and_save(self, name: str, results_dir: str, save_data: bool) -> None:
        if save_data:
            plt.savefig(results_dir + prepare_filename(name))
            plt.close()
        plt.show()