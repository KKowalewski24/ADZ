import numpy as np
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from module.detector.Detector import Detector


class EtsDetector(Detector):

    def detect(self, dataset: np.ndarray) -> None:
        pred = ETSModel(dataset).fit().predict()
