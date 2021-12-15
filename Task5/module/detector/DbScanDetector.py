from typing import Any, Dict

from sklearn.cluster import DBSCAN

from module.detector.Detector import Detector


class DbScanDetector(Detector):

    def detect(self, params: Dict[str, Any]) -> None:
        db_scan = DBSCAN(**params)
