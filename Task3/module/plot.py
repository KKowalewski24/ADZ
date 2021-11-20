import numpy as np
from matplotlib import pyplot as plt

from module.utils import prepare_filename


def draw_plots(X: np.ndarray, y: np.ndarray, name: str, title: str,
               results_dir: str, save_data: bool, size=20) -> None:
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c="k", s=size)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], s=size)

    _set_descriptions(title, "", "")
    _show_and_save(name, results_dir, save_data)


def _set_descriptions(title: str, x_label: str = "", y_label: str = "") -> None:
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)


def _show_and_save(name: str, results_dir: str, save_data: bool) -> None:
    if save_data:
        plt.savefig(results_dir + prepare_filename(name))
        plt.close()
    plt.show()
