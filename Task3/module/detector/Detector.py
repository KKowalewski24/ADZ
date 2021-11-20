from abc import ABC, abstractmethod
from typing import List

import numpy as np


class Detector(ABC):

    @abstractmethod
    def detect(self, dataset: np.ndarray) -> np.ndarray:
        pass


    def get_dataset_logarithm(self, dataset: np.ndarray) -> np.ndarray:
        return np.log(dataset).flatten()


    def fill_outliers_array(self, dataset_size: int, indexes: List[int]) -> np.ndarray:
        outliers = np.zeros(dataset_size)
        np.put(outliers, indexes, 1)
        return outliers
