from typing import Any, Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sesd import seasonal_esd
from sklearn.metrics import precision_score, recall_score

from module.detector.Detector import Detector


class ShesdDetector(Detector):

    def __init__(self, dataset: pd.DataFrame, ground_truth_outliers: np.ndarray,
                 configuration_name: str) -> None:
        super().__init__(dataset, ground_truth_outliers, configuration_name)
        self.outliers: np.ndarray = np.ndarray([])


    def detect(self, params: Dict[str, Any]) -> None:
        outlier_indexes = seasonal_esd(self.dataset.iloc[:, 1], **params)
        self.outliers = self._fill_outliers_array(len(self.dataset.index), outlier_indexes)


    def calculate_statistics(self) -> Dict[str, float]:
        recall = round(recall_score(self.ground_truth_outliers, self.outliers, zero_division=0), 2)
        precision = round(precision_score(
            self.ground_truth_outliers, self.outliers, zero_division=0), 2)

        self.statistics = {
            "recall": recall,
            "precision": precision
        }

        return self.statistics


    def show_results(self, results_dir: str, save_data: bool) -> None:
        plt.figure(figsize=(10, 6))

        # for outlier in self.outliers:
        #     plt.axvline(outlier, alpha=1.0, color="red", linewidth=4)

        plt.plot(self.dataset.iloc[:, 0], self.dataset.iloc[:, 1])
        self._set_descriptions(self.configuration_name + self._statistics_to_string(), "", "")
        self._show_and_save(self.configuration_name, results_dir, save_data)
