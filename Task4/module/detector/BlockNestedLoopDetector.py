from typing import Any, Dict

from sklearn.decomposition import PCA

from module.detector.Detector import Detector


class BlockNestedLoopDetector(Detector):

    def detect(self, params: Dict[str, Any]) -> None:
        pass


    def show_results(self, save_results: bool) -> None:
        pca = PCA(n_components=2)
