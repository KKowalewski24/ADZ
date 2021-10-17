from typing import Union

from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
from skmultiflow.lazy import KNNClassifier
from skmultiflow.trees import HoeffdingTreeClassifier


def classify(detector: BaseDriftDetector,
             classifier: Union[KNNClassifier, HoeffdingTreeClassifier]) -> None:
    pass
