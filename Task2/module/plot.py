from matplotlib import pyplot as plt

from module.utils import create_directory, prepare_filename

RESULTS_DIR = "results/"


def draw_plots(clusterizer_name: str, save_charts: bool) -> None:
    create_directory(RESULTS_DIR)

    # TODO

    plt.title("TODO!!!")
    plt.xlabel("TODO!!!")
    plt.ylabel("TODO!!!")

    if save_charts:
        plt.savefig(RESULTS_DIR + prepare_filename(f"{clusterizer_name}"))
        plt.close()
    plt.show()
