import numpy as np
from sesd import seasonal_esd

from module.detector.Detector import Detector


class ShesdDetector(Detector):

    def detect(self, dataset: np.ndarray) -> None:
        dataset_logarithm = self.get_dataset_logarithm(dataset)
        anomaly_indexes = seasonal_esd(dataset_logarithm)
        for index in anomaly_indexes:
            print(f"index: {index}, value: {dataset_logarithm[index]}")
