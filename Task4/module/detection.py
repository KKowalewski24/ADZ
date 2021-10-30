from typing import Tuple

import numpy as np
from pyod.models.base import BaseDetector


def detect(
        dataset: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        detector: BaseDetector
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test = dataset

    detector.fit(X_train)
    y_train_pred = detector.labels_
    y_train_scores = detector.decision_scores_

    y_test_pred = detector.predict(X_test)
    y_test_scores = detector.decision_function(X_test)

    return y_train_pred, y_test_pred, y_train_scores, y_test_scores
