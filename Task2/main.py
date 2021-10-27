from argparse import ArgumentParser, Namespace
from typing import Dict, List, Union

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.neighbors import LocalOutlierFactor

from module.LatexGenerator import LatexGenerator
from module.analysis import clusterize
from module.reader import read_dataset_1, read_dataset_2, read_dataset_3
from module.utils import create_directory, display_finish, prepare_filename, run_main

"""
    How to run:
        python main.py -s -c kmeans
"""

# VAR ------------------------------------------------------------------------ #
RESULTS_DIR = "results/"

CLUSTERIZERS_SETUP: Dict = {
    "kmeans": KMeans,
    "agglomerative": AgglomerativeClustering
}

BENCHMARK_ALGORITHMS_SETUP: Dict = {
    "db_scan": DBSCAN(),
    "lof": LocalOutlierFactor()
}

latex_generator: LatexGenerator = LatexGenerator(RESULTS_DIR)


# MAIN ----------------------------------------------------------------------- #
def main() -> None:
    args = prepare_args()
    chosen_clusterizer_name = args.clusterizer
    save_stats = args.save
    create_directory(RESULTS_DIR)

    print("Reading datasets ...")
    datasets: Dict[str, pd.DataFrame] = {
        "dataset_1": read_dataset_1(),
        "dataset_2": read_dataset_2(),
        "dataset_3": read_dataset_3(),
    }

    for dataset in datasets:
        print(f"Clustering dataset: {dataset} ...")
        statistics_list = clusterize(
            datasets[dataset], CLUSTERIZERS_SETUP[chosen_clusterizer_name](),
            list(BENCHMARK_ALGORITHMS_SETUP.values())
        )

        if save_stats:
            print(f"Saving results to file for dataset: {dataset} ...")
            benchmark_stats = [
                convert_statistics(statistics_list[i], list(BENCHMARK_ALGORITHMS_SETUP.keys())[i - 1])
                for i in range(1, len(statistics_list))
            ]

            latex_generator.generate_vertical_table(
                ["Classifier", "Silhouette", "Calinski_Harabasz",
                 "Davies_Bouldin", "Rand_score", "Fowlkes_Mallows"],
                [convert_statistics(statistics_list[0], chosen_clusterizer_name)] + benchmark_stats,
                dataset + "_metrics"
            )

    display_finish()


# DEF ------------------------------------------------------------------------ #
def convert_statistics(statistics: Dict[str, float], algorithm_name: str) -> List[Union[str, float]]:
    return [
        algorithm_name,
        statistics["silhouette"],
        statistics["calinski_harabasz"],
        statistics["davies_bouldin"],
        statistics["rand_score"],
        statistics["fowlkes_mallows"]
    ]


def draw_plots(clusterizer_name: str, save_charts: bool, results_dir: str) -> None:
    # TODO
    plt.title("TODO!!!")
    plt.xlabel("TODO!!!")
    plt.ylabel("TODO!!!")

    if save_charts:
        plt.savefig(results_dir + prepare_filename(f"{clusterizer_name}"))
        plt.close()
    plt.show()


def prepare_args() -> Namespace:
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        "-c", "--clusterizer", type=str, choices=CLUSTERIZERS_SETUP.keys(),
        help="Name of clusterizer"
    )
    arg_parser.add_argument(
        "-s", "--save", default=False, action="store_true", help="Save charts to files"
    )

    return arg_parser.parse_args()


# __MAIN__ ------------------------------------------------------------------- #
if __name__ == "__main__":
    run_main(main)
