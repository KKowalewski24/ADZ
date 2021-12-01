from typing import Any, Dict

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from module.detector.Detector import Detector


class ArimaDetector(Detector):

    def __init__(self, dataset: pd.DataFrame, ground_truth_outliers: np.ndarray,
                 configuration_name: str, ) -> None:
        super().__init__(dataset, ground_truth_outliers, configuration_name)


    def detect(self, params: Dict[str, Any]) -> None:
        threshold = params["threshold"]
        del params["threshold"]
        x = self.dataset.iloc[:, 1]
        model = ARIMA(x, **params).fit()
        err = model.resid ** 2
        self.outliers_array = np.array(err > threshold * err.std(), dtype=np.int8)
        self.outlier_indexes = np.argwhere(self.outliers_array).squeeze()
