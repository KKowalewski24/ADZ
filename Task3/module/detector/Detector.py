from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score

from module.utils import prepare_filename


class Detector(ABC):

    def __init__(self, dataset: pd.DataFrame, ground_truth_outliers: np.ndarray,
                 configuration_name: str) -> None:
        self.dataset = dataset
        self.ground_truth_outliers = ground_truth_outliers
        self.configuration_name = configuration_name
        self.statistics: Dict[str, float] = {}
        self.outliers_array: np.ndarray = np.ndarray([])
        self.outlier_indexes: List = []


    @abstractmethod
    def detect(self, params: Dict[str, Any]) -> None:
        pass


    def calculate_statistics(self) -> Dict[str, float]:
        recall = round(recall_score(
            self.ground_truth_outliers, self.outliers_array, zero_division=0), 2)
        precision = round(precision_score(
            self.ground_truth_outliers, self.outliers_array, zero_division=0), 2)

        self.statistics = {
            "recall": recall,
            "precision": precision
        }

        return self.statistics


    def show_results(self, results_dir: str, save_data: bool) -> None:
        plt.figure(figsize=(10, 6))
        for index in sorted(self.outlier_indexes):
            plt.plot(
                self.dataset.iloc[index, 0], self.dataset.iloc[index, 1], "ro"
            )
        for index in sorted(np.argwhere(self.ground_truth_outliers).squeeze()):
            plt.plot(
                self.dataset.iloc[index, 0], self.dataset.iloc[index, 1], "gx", markersize=12
            )

        plt.plot(self.dataset.iloc[:, 0], self.dataset.iloc[:, 1])
        self._set_descriptions(
            self.configuration_name + self._statistics_to_string(),
            self.dataset.columns[0], self.dataset.columns[1]
        )
        self._show_and_save(self.configuration_name, results_dir, save_data)


    def _fill_outliers_array(self, dataset_size: int, indexes: List[int]) -> np.ndarray:
        outliers_array = np.zeros(dataset_size)
        np.put(outliers_array, indexes, 1)
        return outliers_array


    def _statistics_to_string(self) -> str:
        return " ".join([stat + '=' + str(self.statistics[stat]) for stat in self.statistics])


    def _set_descriptions(self, title: str, x_label: str = "", y_label: str = "") -> None:
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)


    def _show_and_save(self, name: str, results_dir: str, save_data: bool) -> None:
        if save_data:
            plt.savefig(results_dir + prepare_filename(name))
            plt.close()
        plt.show()
