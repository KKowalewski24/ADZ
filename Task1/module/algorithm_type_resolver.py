from typing import List

from skmultiflow.drift_detection import ADWIN, DDM, HDDM_A, KSWIN, PageHinkley
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
from skmultiflow.lazy import KNNClassifier


def resolve_detector_type(detector_names: List[str], chosen_detector_name: str) -> BaseDriftDetector:
    # TODO Consider setting different params
    if detector_names[0] == chosen_detector_name:
        return ADWIN()
    elif detector_names[1] == chosen_detector_name:
        return DDM()
    elif detector_names[2] == chosen_detector_name:
        return HDDM_A()
    elif detector_names[3] == chosen_detector_name:
        return KSWIN()
    elif detector_names[4] == chosen_detector_name:
        return PageHinkley()


def resolve_classifier_type(classifier_names: List[str], chosen_classifier_name: str):
    if classifier_names[0] == chosen_classifier_name:
        # TODO Consider different knn params
        return KNNClassifier(n_neighbors=8, max_window_size=2000)
