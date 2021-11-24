from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from module.detector.Detector import Detector


class EtsDetector(Detector):

    def __init__(self, dataset: np.ndarray, ground_truth_outliers: np.ndarray,
                 configuration_name: str) -> None:
        super().__init__(dataset, ground_truth_outliers, configuration_name)
        self.ets_model = None


    def detect(self, params: Dict[str, Any]) -> None:
        dataset_logarithm = self._calculate_dataset_logarithm()
        self.ets_model = ETSModel(dataset_logarithm, **params).fit()


    def calculate_statistics(self) -> Dict[str, float]:
        self.statistics = {
            "aic": round(self.ets_model.aic, 2),
            "bic": round(self.ets_model.bic, 2),
            "hqic": round(self.ets_model.hqic, 2),
        }

        return self.statistics


    def show_results(self, results_dir: str, save_data: bool) -> None:
        plt.figure(figsize=(10, 6))
        plt.plot(self.dataset)
        plt.plot(self.ets_model.fittedvalues)
        plt.legend()
        self._set_descriptions(self.configuration_name + self._statistics_to_string(), "", "")
        self._show_and_save(self.configuration_name, results_dir, save_data)
