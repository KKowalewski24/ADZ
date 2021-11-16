from typing import Dict

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from module.utils import prepare_filename

colors: Dict[int, str] = {
    -1: "k",
    0: "b",
    1: "g",
    2: "r",
    3: "c",
    4: "m",
    5: "y",
    6: "w",
}


def draw_plots(df: pd.DataFrame, name: str, results_dir: str, save_data: bool) -> None:
    reduced_data = PCA(n_components=2).fit_transform(df)
    plt.scatter(
        reduced_data[:, 0], reduced_data[:, 1], color=df.cluster.map(colors), s=15.0
    )

    # TODO Check if PCA works as expected
    # plt.scatter(
    #     df.iloc[:, 0], df.iloc[:, 1], color=df.cluster.map(colors), s=15.0
    # )

    _set_descriptions(name, "", "")
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
