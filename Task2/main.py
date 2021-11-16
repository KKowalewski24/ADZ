from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List

import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.neighbors import LocalOutlierFactor

from module.LatexGenerator import LatexGenerator
from module.plot import draw_plots
from module.reader import read_synthetic_dataset
from module.utils import create_directory, display_finish, run_main

"""
    How to run:
        python main.py -s -c lof
"""

# VAR ------------------------------------------------------------------------ #
RESULTS_DIR = "results/"
latex_generator: LatexGenerator = LatexGenerator(RESULTS_DIR)

clusterizers: Dict[str, Any] = {
    "kmeans": KMeans,
    "agglomerative": AgglomerativeClustering,
    "db_scan": DBSCAN,
    "lof": LocalOutlierFactor
}

datasets: Dict[str, pd.DataFrame] = {
    "synthetic": read_synthetic_dataset(),
}


# MAIN ----------------------------------------------------------------------- #
def main() -> None:
    args = prepare_args()
    chosen_clusterizer_name = args.clusterizer
    chosen_dataset_name = args.dataset
    algorithm_params = args.algorithm_params
    save_stats = args.save
    create_directory(RESULTS_DIR)

    clusters: List[int] = (clusterizers[chosen_clusterizer_name]()
                           .fit_predict(datasets[chosen_dataset_name]))

    display_finish()


# DEF ------------------------------------------------------------------------ #
def prepare_args() -> Namespace:
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        "-c", "--clusterizer", type=str, choices=clusterizers.keys(),
        help="Name of clusterizer"
    )
    arg_parser.add_argument(
        "-ds", "--dataset", type=str, choices=datasets.keys(),
        help="Name of dataset"
    )
    arg_parser.add_argument(
        "-ap", "--algorithm_params", nargs="+", required=True,
        help="List of arguments for certain algorithm"
    )
    arg_parser.add_argument(
        "-s", "--save", default=False, action="store_true", help="Save charts to files"
    )

    return arg_parser.parse_args()


# __MAIN__ ------------------------------------------------------------------- #
if __name__ == "__main__":
    run_main(main)
