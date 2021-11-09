import pandas as pd
from matplotlib import pyplot as plt

from module.utils import prepare_filename


def draw_plots(x_axis_data: pd.Series, y_axis_data: pd.Series, radius: float,
               name: str, results_dir: str, save_data: bool) -> None:
    plt.scatter(
        x_axis_data, y_axis_data, color="k", s=3.0, label="Data points"
    )
    plt.scatter(
        x_axis_data, y_axis_data, s=1000 * radius,
        edgecolors="r", facecolors="none", label="Outlier scores"
    )
    _set_descriptions("", "", "")
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
