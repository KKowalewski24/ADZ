import numpy as np
from sesd import seasonal_esd

from module.detector.Detector import Detector


class ShesdDetector(Detector):

    def detect(self, dataset: np.ndarray) -> None:
        anomalies = seasonal_esd(dataset)
