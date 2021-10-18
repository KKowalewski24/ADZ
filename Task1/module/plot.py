from typing import List

import seaborn
from matplotlib import pyplot as plt

from module.utils import create_directory, prepare_filename

RESULTS_DIR = "results/"


def draw_plots(changes: List[int], warnings: List[int], accuracy_trend: List[float],
               train_size_range: List[int], detector_name: str, classifier_name: str,
               save_charts: bool) -> None:
    create_directory(RESULTS_DIR)
    # TODO CONSIDER DIFFERENT COLORS
    for change in changes:
        plt.axvline(change, alpha=0.3, color="red")

    for warning in warnings:
        plt.axvline(warning, alpha=0.3, color="yellow")

    seaborn.lineplot(x=train_size_range, y=accuracy_trend, alpha=0.4, color="black")

    plt.title(f"{detector_name}_{classifier_name}")
    # TODO ADD PROPER LABELS
    plt.xlabel("TODO!!!")
    plt.ylabel("TODO!!!")
    if save_charts:
        plt.savefig(RESULTS_DIR + prepare_filename(f"{detector_name}_{classifier_name}"))
        plt.close()
    plt.show()