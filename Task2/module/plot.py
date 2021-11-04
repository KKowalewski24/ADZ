from matplotlib import pyplot as plt

from module.utils import prepare_filename


def draw_plots(clusterizer_name: str, results_dir: str, save_data: bool) -> None:
    # TODO

    _set_descriptions("", "", "")
    _show_and_save(clusterizer_name, results_dir, save_data)


def _set_descriptions(title: str, x_label: str = "", y_label: str = "") -> None:
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)


def _show_and_save(name: str, results_dir: str, save_data: bool) -> None:
    if save_data:
        plt.savefig(results_dir + prepare_filename(name))
        plt.close()
    plt.show()
