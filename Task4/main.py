from argparse import ArgumentParser, Namespace
from typing import Dict, Tuple

import numpy as np
from pyod.models.abod import ABOD
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize

from module.detection import detect
from module.reader import read_dataset_1, read_dataset_2, read_dataset_3
from module.utils import create_directory, display_finish, run_main

"""
    How to run: python main.py -s -a abod
        
"""

# VAR ------------------------------------------------------------------------ #
RESULTS_DIR = "results/"

ALGORITHM_SETUP: Dict[str, Tuple] = {
    "abod": (ABOD, {"method": "default"}),
    "fast_abod": (ABOD, {"method": "fast"})
}


# MAIN ----------------------------------------------------------------------- #
def main() -> None:
    args = prepare_args()
    chosen_algorithm_name = args.algorithm
    save_results = args.save
    create_directory(RESULTS_DIR)

    print("Reading datasets ...")
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {
        "dataset_1": read_dataset_1(),
        "dataset_2": read_dataset_2(),
        "dataset_3": read_dataset_3(),
    }

    for dataset in datasets:
        print(f"Detecting dataset: {dataset} ...")
        y_train_pred, y_test_pred, y_train_scores, y_test_scores = detect(
            datasets[dataset],
            ALGORITHM_SETUP[chosen_algorithm_name][0](
                **ALGORITHM_SETUP[chosen_algorithm_name][1]
            )
        )

        X_train, X_test, y_train, y_test = datasets[dataset]
        print("\nOn Training Data:")
        evaluate_print(chosen_algorithm_name, y_train, y_train_scores)
        print("\nOn Test Data:")
        evaluate_print(chosen_algorithm_name, y_test, y_test_scores)

        visualize(
            chosen_algorithm_name, X_train, y_train, X_test, y_test, y_train_pred,
            y_test_pred, show_figure=not save_results, save_figure=save_results
        )

        if save_results:
            pass

    display_finish()


# DEF ------------------------------------------------------------------------ #
def prepare_args() -> Namespace:
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        "-a", "--algorithm", type=str, choices=ALGORITHM_SETUP.keys(), help="Name of algorithm"
    )
    arg_parser.add_argument(
        "-s", "--save", default=False, action="store_true", help="Save results to files"
    )

    return arg_parser.parse_args()


# __MAIN__ ------------------------------------------------------------------- #
if __name__ == "__main__":
    run_main(main)
