from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from module.detector.Detector import Detector


class EtsDetector(Detector):

    def __init__(self, dataset: pd.DataFrame, ground_truth_outliers: np.ndarray,
                 configuration_name: str) -> None:
        super().__init__(dataset, ground_truth_outliers, configuration_name)
        self.ets_model = None


    def detect(self, params: Dict[str, Any]) -> None:
        self.ets_model = ETSModel(self.dataset.squeeze(), **params).fit()


    def calculate_statistics(self) -> Dict[str, float]:
        self.statistics = {
            "aic": round(self.ets_model.aic, 2),
            "bic": round(self.ets_model.bic, 2),
            "hqic": round(self.ets_model.hqic, 2),
        }

        return self.statistics


    def show_results(self, results_dir: str, save_data: bool) -> None:
        plt.figure(figsize=(10, 6))
        plt.plot(self.dataset, label="Original data")
        plt.plot(self.ets_model.fittedvalues, label="Predicted")
        plt.legend()
        self._set_descriptions(self.configuration_name + self._statistics_to_string(), "", "")
        self._show_and_save(self.configuration_name, results_dir, save_data)
