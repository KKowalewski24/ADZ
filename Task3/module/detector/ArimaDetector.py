from typing import Any, Dict

from statsmodels.tsa.arima.model import ARIMA

from module.detector.Detector import Detector


class ArimaDetector(Detector):

    def detect(self, params: Dict[str, Any]) -> None:
        dataset_logarithm = self._calculate_dataset_logarithm()
        pred = ARIMA(dataset_logarithm, **params).fit().predict()
        print(pred)


    def calculate_statistics(self) -> Dict[str, float]:
        pass


    def show_results(self, results_dir: str, save_data: bool) -> None:
        pass
