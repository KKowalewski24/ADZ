from typing import List

from skmultiflow.drift_detection import ADWIN, DDM, HDDM_A, KSWIN, PageHinkley
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector


def resolve_detector_type(detector_names: List[str], chosen_detector_name: str) -> BaseDriftDetector:
    if detector_names[0] == chosen_detector_name:
        return ADWIN()
    elif detector_names[1] == chosen_detector_name:
        return DDM()
    elif detector_names[2] == chosen_detector_name:
        return HDDM_A()
    elif detector_names[3] == chosen_detector_name:
        return KSWIN(alpha=0.1)
    elif detector_names[4] == chosen_detector_name:
        return PageHinkley()
