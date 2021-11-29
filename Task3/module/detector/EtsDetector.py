from typing import Any, Dict

import numpy as np
import pandas as pd

from module.detector.Detector import Detector


class EtsDetector(Detector):

    def __init__(self, dataset: pd.DataFrame, ground_truth_outliers: np.ndarray,
                 configuration_name: str) -> None:
        super().__init__(dataset, ground_truth_outliers, configuration_name)


    def detect(self, params: Dict[str, Any]) -> None:
        # series = pd.Series(
        #     self.dataset.iloc[:, 1].to_numpy().astype(float),
        #     index=self.dataset.iloc[:, 0]
        # )
        # ETSModel(series, **params).fit()
        pass
