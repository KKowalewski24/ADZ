from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.neighbors import LocalOutlierFactor

from module.LatexGenerator import LatexGenerator
from module.analysis import clusterize
from module.plot import draw_plots
from module.reader import read_iris_ds, read_penguins_dataset, read_synthetic_dataset, \
    read_wheat_seeds_dataset
from module.utils import create_directory, display_finish, run_main

"""
    How to run:
        python main.py -s -c lof
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
    # TODO SET PROPER DIMENSIONS TO DRAW
    datasets: Dict[str, Tuple[pd.DataFrame, List[Tuple[int, int]]]] = {
        "penguins": (read_penguins_dataset(), [(0, 1)]),
        "wheat_seeds": (read_wheat_seeds_dataset(), [(0, 1)]),
        "synthetic_dataset": (read_synthetic_dataset(), [(0, 1)]),
        "iris": (read_iris_ds(), [(0, 1)]),
    }

    for dataset in datasets:
        print(f"Clustering dataset: {dataset} ...")
        current_dataset = datasets[dataset][0]
        dimensions_to_draw = datasets[dataset][1]

        for params in CLUSTERIZERS_SETUP[chosen_clusterizer_name][1]:
            radius = clusterize(
                current_dataset,
                CLUSTERIZERS_SETUP[chosen_clusterizer_name][0](**params)
            )
            for dimension in dimensions_to_draw:
                name = (
                    f"{dataset}_{chosen_clusterizer_name}_"
                    f"{'_'.join([str(params[param]) for param in params])}_"
                    f"{'_'.join([str(dim) for dim in dimension])}"
                )
                draw_plots(
                    current_dataset.iloc[:, dimension[0]],
                    current_dataset.iloc[:, dimension[1]],
                    radius, name, RESULTS_DIR, save_stats
                )

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
