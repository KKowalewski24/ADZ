from abc import ABC, abstractmethod

import numpy as np


class Detector(ABC):

    @abstractmethod
    def detect(self, dataset: np.ndarray) -> None:
        pass
