from typing import Dict

from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from module.detector.Detector import Detector


class EtsDetector(Detector):

    def detect(self) -> None:
        dataset_logarithm = self._calculate_dataset_logarithm()
        pred = ETSModel(dataset_logarithm).fit().predict()
        print(pred)


    def calculate_statistics(self) -> Dict[str, float]:
        pass


    def show_results(self, results_dir: str, save_data: bool) -> None:
        pass
