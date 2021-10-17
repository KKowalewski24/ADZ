from typing import List

import seaborn
from matplotlib import pyplot as plt

from module.utils import create_directory, prepare_filename

RESULTS_DIR = "results/"


def draw_plots(training_size_range: List[int], accuracy_trend: List[float], warnings: [],
               changes: [], detector_name: str, classifier_name: str, save_charts: bool) -> None:
    create_directory(RESULTS_DIR)
    seaborn.lineplot(x=training_size_range, y=accuracy_trend, alpha=0.4, color="green")

    for change in changes:
        plt.axvline(change, alpha=0.3, color="red")

    for warning in warnings:
        plt.axvline(warning, alpha=0.3, color="yellow")

    plt.title(f"{detector_name}_{classifier_name}")
    plt.xlabel("TODO!!!")
    plt.ylabel("TODO!!!")
    if save_charts:
        plt.savefig(RESULTS_DIR + prepare_filename(f"{detector_name}_{classifier_name}"))
    plt.show()
