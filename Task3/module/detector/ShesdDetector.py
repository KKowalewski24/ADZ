from typing import Any, Dict

import numpy as np
import pandas as pd
from sesd import seasonal_esd

from module.detector.Detector import Detector


class ShesdDetector(Detector):

    def __init__(self, dataset: pd.DataFrame, ground_truth_outliers: np.ndarray,
                 configuration_name: str) -> None:
        super().__init__(dataset, ground_truth_outliers, configuration_name)


    def detect(self, params: Dict[str, Any]) -> None:
        self.outlier_indexes = seasonal_esd(self.dataset.iloc[:, 1], **params)
        self.outliers_array = self._fill_outliers_array(len(self.dataset.index), self.outlier_indexes)
