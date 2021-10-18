from matplotlib import pyplot as plt

from module.utils import prepare_filename


def draw_plots(clusterizer_name: str, save_charts: bool, results_dir: str) -> None:
    # TODO
    plt.title("TODO!!!")
    plt.xlabel("TODO!!!")
    plt.ylabel("TODO!!!")

    if save_charts:
        plt.savefig(results_dir + prepare_filename(f"{clusterizer_name}"))
        plt.close()
    plt.show()

