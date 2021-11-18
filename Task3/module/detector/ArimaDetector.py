import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from module.detector.Detector import Detector


class ArimaDetector(Detector):

    def detect(self, dataset: np.ndarray) -> None:
        pred = ARIMA(dataset, order=(1, 1, 2)).fit().predict()
