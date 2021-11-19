from abc import ABC, abstractmethod

import numpy as np


class Detector(ABC):

    @abstractmethod
    def detect(self, dataset: np.ndarray) -> None:
        pass


    def get_dataset_logarithm(self, dataset: np.ndarray) -> np.ndarray:
        return np.log(dataset).flatten()
