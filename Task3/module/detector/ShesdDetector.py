import numpy as np
from sesd import seasonal_esd

from module.detector.Detector import Detector


class ShesdDetector(Detector):

    def detect(self, dataset: np.ndarray) -> np.ndarray:
        dataset_logarithm = self.get_dataset_logarithm(dataset)
        outlier_indexes = seasonal_esd(dataset_logarithm)
        return self.fill_outliers_array(dataset.size, outlier_indexes)
