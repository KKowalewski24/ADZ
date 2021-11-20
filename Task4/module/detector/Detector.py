from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class Detector(ABC):

    @abstractmethod
    def detect(self, dataset: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
        pass
