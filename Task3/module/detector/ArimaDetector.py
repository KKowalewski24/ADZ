import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from module.detector.Detector import Detector


class ArimaDetector(Detector):

    def detect(self, dataset: np.ndarray) -> None:
        pred = ARIMA(self.get_dataset_logarithm(dataset), order=(2, 1, 2)).fit().predict()
        print(pred)
