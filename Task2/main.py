from argparse import ArgumentParser, Namespace
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import precision_score, recall_score
from sklearn.neighbors import LocalOutlierFactor

from module.LatexGenerator import LatexGenerator
from module.plot import draw_plots
from module.reader import read_http_dataset, read_mammography_dataset, read_synthetic_dataset
from module.utils import create_directory, display_finish, run_main
from module.OutlierKMeans import OutlierKMeans
from module.OutlierAgglomerativeClustering import OutlierAgglomerativeClustering

"""
    How to run:
        python main.py -s -c lof
"""

# VAR ------------------------------------------------------------------------ #
RESULTS_DIR = "results/"
latex_generator: LatexGenerator = LatexGenerator(RESULTS_DIR)

clusterizers: Dict[str, Any] = {
    "kmeans": (OutlierKMeans, int, float),
    "agglomerative": (OutlierAgglomerativeClustering, float, float),
    "db_scan": (DBSCAN, float),
    "lof": (LocalOutlierFactor, int)
}

datasets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
    "http": read_http_dataset(),
    "mammography": read_mammography_dataset(),
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

    X, y = datasets[chosen_dataset_name]
    params = [typee(param) for param, typee in zip(algorithm_params, clusterizers[chosen_clusterizer_name][1:])]
    y_pred = clusterizers[chosen_clusterizer_name][0](*params).fit_predict(X)
    recall = np.round(recall_score(y, y_pred, average=None, zero_division=0)[0], 2)
    precision = np.round(precision_score(y, y_pred, average=None, zero_division=0)[0], 2)

    print(f"Recall {recall} & Precision {precision}")
    name = (f"{chosen_clusterizer_name}_{chosen_dataset_name}_"
            f"{'_'.join([str(param) for param in algorithm_params])}_")
    title = name + f"Rcl={recall}_Prec={precision}"
    draw_plots(X, y_pred, name, title, RESULTS_DIR, save_stats)

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
        "-ap", "--algorithm_params", nargs="+", required=True, type=str,
        help="List of arguments for certain algorithm"
    )
    arg_parser.add_argument(
        "-s", "--save", default=False, action="store_true", help="Save charts to files"
    )

    return arg_parser.parse_args()


# __MAIN__ ------------------------------------------------------------------- #
if __name__ == "__main__":
    run_main(main)
