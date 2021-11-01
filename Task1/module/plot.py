from typing import Any, Dict, List

import seaborn
from matplotlib import pyplot as plt

from module.utils import create_directory, prepare_filename

RESULTS_DIR = "results/"


def draw_plots(changes: List[int], warnings: List[int],
               accuracy_trend: List[float], train_size_range: List[int],
               detector_name: str, params: Dict[str, Any], save_charts: bool) -> None:
    create_directory(RESULTS_DIR)

    for change in changes:
        plt.axvline(change, alpha=0.3, color="red")

    for warning in warnings:
        plt.axvline(warning, alpha=0.3, color="orange")

    seaborn.lineplot(x=train_size_range, y=accuracy_trend, alpha=0.4, color="green")

    plt.title(
        f"KNN {detector_name} {' '.join([param + '=' + str(params[param]) for param in params])}"
    )
    plt.xlabel("Numer próbki")
    plt.ylabel("Dokładność")
    if save_charts:
        plt.savefig(
            RESULTS_DIR + prepare_filename(
                f"KNN_{detector_name}_{'_'.join([str(params[param]) for param in params])}")
        )
        plt.close()
    plt.show()
