from typing import List

from sesd import seasonal_esd
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel


def resolve_pattern_detector(algorithm_names: List[str], chosen_algorithm: str):
    # TODO Consider setting different params
    if algorithm_names[0] == chosen_algorithm:
        return ARIMA()
    elif algorithm_names[1] == chosen_algorithm:
        return ETSModel()
    elif algorithm_names[2] == chosen_algorithm:
        return seasonal_esd()
