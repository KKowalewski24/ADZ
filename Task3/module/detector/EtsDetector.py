import numpy as np
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from module.detector.Detector import Detector


class EtsDetector(Detector):

    def detect(self, dataset: np.ndarray) -> None:
        dataset_logarithm = self.get_dataset_logarithm(dataset)
        pred = ETSModel(dataset_logarithm).fit().predict()
        print(pred)
