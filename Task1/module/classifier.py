from typing import List, Tuple

from skmultiflow.data import FileStream
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
from skmultiflow.lazy import KNNClassifier
from tqdm import tqdm


def classify(
        detector: BaseDriftDetector, dataset: FileStream, window_size: int
) -> Tuple[List[int], List[int], List[float], List[int]]:
    classifier = KNNClassifier(n_neighbors=5, leaf_size=100, max_window_size=window_size)
    warning_zone_samples = ([], [])
    changes: List[int] = []
    warnings: List[int] = []
    accuracy_trend: List[float] = []
    correct_predictions: int = 0

    for i in tqdm(range(1, window_size + 1)):
        X, y = dataset.next_sample()
        pred = classifier.predict(X)

        if pred == y:
            detector.add_element(0)
            correct_predictions += 1
        else:
            detector.add_element(1)

        if detector.detected_change():
            classifier.reset()
            if len(warning_zone_samples[0]) > 0:
                classifier.fit(warning_zone_samples[0], warning_zone_samples[1])
                warning_zone_samples = ([], [])
            changes.append(i)

        if detector.detected_warning_zone():
            warnings.append(i)
            warning_zone_samples[0].append(X)
            warning_zone_samples[1].append(y)
        else:
            warning_zone_samples = ([], [])

        classifier.partial_fit(X, y)
        accuracy_trend.append((correct_predictions / i) * 100)

    return changes, warnings, accuracy_trend, list(range(window_size))
