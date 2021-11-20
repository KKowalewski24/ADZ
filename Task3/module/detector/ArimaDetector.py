import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from module.detector.Detector import Detector


class ArimaDetector(Detector):

    def detect(self, dataset: np.ndarray) -> np.ndarray:
        dataset_logarithm = self.get_dataset_logarithm(dataset)
        pred = ARIMA(dataset_logarithm, order=(2, 1, 2)).fit().predict()
        print(pred)
