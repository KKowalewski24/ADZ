from argparse import ArgumentParser, Namespace
from typing import Dict, List, Union

import pandas as pd
from matplotlib import pyplot as plt

from module.LatexGenerator import LatexGenerator
from module.algorithm_type_resolver import prepare_benchmark_algorithms, resolve_clusterizer_type
from module.analysis import clusterize
from module.reader import read_dataset_1, read_dataset_2, read_dataset_3
from module.utils import check_if_exists_in_args, check_types_check_style, compile_to_pyc, \
    create_directory, display_finish, prepare_filename

"""
    How to run:
        python main.py -s -c kmeans
"""

# VAR ------------------------------------------------------------------------ #
RESULTS_DIR = "results/"
CLUSTERIZER_NAMES: List[str] = ["kmeans", "agglomerative"]

latex_generator: LatexGenerator = LatexGenerator(RESULTS_DIR)


# MAIN ----------------------------------------------------------------------- #
def main() -> None:
    args = prepare_args()
    chosen_clusterizer_name = args.clusterizer
    save_stats = args.save
    create_directory(RESULTS_DIR)

    clusterizer = resolve_clusterizer_type(CLUSTERIZER_NAMES, chosen_clusterizer_name)
    benchmark_algorithms, benchmark_algorithms_names = prepare_benchmark_algorithms()
    print("Reading datasets ...")
    datasets: Dict[str, pd.DataFrame] = {
        "dataset_1": read_dataset_1(),
        "dataset_2": read_dataset_2(),
        "dataset_3": read_dataset_3(),
    }

    for dataset in datasets:
        print(f"Clustering dataset: {dataset} ...")
        statistics_list = clusterize(datasets[dataset], clusterizer, benchmark_algorithms)
        if save_stats:
            print(f"Saving results to file for dataset: {dataset} ...")
            benchmark_stats = [
                convert_statistics(statistics_list[i], benchmark_algorithms_names[i - 1])
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
        "-c", "--clusterizer", type=str, choices=CLUSTERIZER_NAMES,
        help="Name of clusterizer"
    )
    arg_parser.add_argument(
        "-s", "--save", default=False, action="store_true", help="Save charts to files"
    )

    return arg_parser.parse_args()


# __MAIN__ ------------------------------------------------------------------- #
if __name__ == "__main__":
    if check_if_exists_in_args("-t"):
        check_types_check_style()
    elif check_if_exists_in_args("-b"):
        compile_to_pyc()
    else:
        main()