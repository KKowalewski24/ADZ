from typing import Any, Dict

import numpy as np
import pandas as pd
from sesd import seasonal_esd, generalized_esd

from module.detector.Detector import Detector


class ShesdDetector(Detector):
    def __init__(
        self,
        dataset: pd.DataFrame,
        ground_truth_outliers: np.ndarray,
        configuration_name: str,
    ) -> None:
        super().__init__(dataset, ground_truth_outliers, configuration_name)

    def detect(self, params: Dict[str, Any]) -> None:
        if "periodicity" in params:
            self.outlier_indexes = seasonal_esd(self.dataset.iloc[:, 1], **params)
        else:
            self.outlier_indexes = generalized_esd(self.dataset.iloc[:, 1].diff(), **params)
        self.outliers_array = self._fill_outliers_array(
            len(self.dataset.index), self.outlier_indexes
        )
