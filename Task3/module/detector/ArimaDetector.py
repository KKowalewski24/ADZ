import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

from module.detector.Detector import Detector


class ArimaDetector(Detector):

    def detect(self, dataset: np.ndarray) -> None:
        dataset_log = np.log(dataset)
        plt.plot(dataset_log)
        plt.show()
        pred = ARIMA(dataset_log, order=(2, 1, 2)).fit().predict()
        print(pred)
