from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np


class Detector(ABC):
    RESULTS_DIR = "results/"


    def __init__(self, dataset: Tuple[np.ndarray, np.ndarray], configuration_name: str) -> None:
        self.dataset = dataset
        self.configuration_name = configuration_name


    @abstractmethod
    def detect(self, params: Dict[str, Any]) -> None:
        pass


    @abstractmethod
    def show_results(self, save_results: bool) -> None:
        pass
