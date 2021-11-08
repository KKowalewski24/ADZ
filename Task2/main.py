from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.neighbors import LocalOutlierFactor

from module.LatexGenerator import LatexGenerator
from module.analysis import clusterize
from module.reader import read_dataset_2, read_dataset_3, read_dataset_penguins, read_iris_ds
from module.utils import create_directory, display_finish, run_main

"""
    How to run:
        python main.py -s -c kmeans
"""

# VAR ------------------------------------------------------------------------ #
RESULTS_DIR = "results/"
latex_generator: LatexGenerator = LatexGenerator(RESULTS_DIR)

CLUSTERIZERS_SETUP: Dict[str, Tuple[Any, List[Dict[str, Any]]]] = {
    "kmeans": (KMeans, [{}]),
    "agglomerative": (AgglomerativeClustering, [{}]),
    "db_scan": (DBSCAN, [{}]),
    "lof": (LocalOutlierFactor, [{}])
}


# MAIN ----------------------------------------------------------------------- #
def main() -> None:
    args = prepare_args()
    chosen_clusterizer_name = args.clusterizer
    save_stats = args.save
    create_directory(RESULTS_DIR)

    print("Reading datasets ...")
    datasets: Dict[str, pd.DataFrame] = {
        "penguins": read_dataset_penguins(),
        "dataset_2": read_dataset_2(),
        "dataset_3": read_dataset_3(),
        "iris": read_iris_ds(),
    }

    for dataset in datasets:
        print(f"Clustering dataset: {dataset} ...")
        for params in CLUSTERIZERS_SETUP[chosen_clusterizer_name][1]:
            clusterize(
                datasets[dataset],
                CLUSTERIZERS_SETUP[chosen_clusterizer_name][0](**params)
            )

        if save_stats:
            print(f"Saving results to file for dataset: {dataset} ...")

    display_finish()


# DEF ------------------------------------------------------------------------ #
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
