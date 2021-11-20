from argparse import ArgumentParser, Namespace
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import precision_score, recall_score

from module.detector.ArimaDetector import ArimaDetector
from module.detector.Detector import Detector
from module.detector.EtsDetector import EtsDetector
from module.detector.ShesdDetector import ShesdDetector
from module.plot import draw_plots
from module.reader import read_air_passengers, read_dataset_2, read_dataset_3
from module.utils import create_directory, display_finish, run_main

"""
    How to run:
        python main.py -s -ds air_passengers -d shesd -ap 1
"""

# VAR ------------------------------------------------------------------------ #
RESULTS_DIR = "results/"

DETECTORS: Dict[str, Detector] = {
    "arima": ArimaDetector(),
    "ets": EtsDetector(),
    "shesd": ShesdDetector()
}

DATASETS: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
    "air_passengers": read_air_passengers(),
    "dataset_2": read_dataset_2(),
    "dataset_3": read_dataset_3(),
}


# MAIN ----------------------------------------------------------------------- #
def main() -> None:
    args = prepare_args()
    chosen_detector_name = args.detector
    chosen_dataset_name = args.dataset
    algorithm_params = args.algorithm_params
    save_stats = args.save
    create_directory(RESULTS_DIR)

    detector: Detector = DETECTORS[chosen_detector_name]
    X, y = DATASETS[chosen_dataset_name]
    outliers = detector.detect(X)

    recall = round(recall_score(y, outliers, zero_division=0), 2)
    precision = round(precision_score(y, outliers, zero_division=0), 2)
    print(f"Recall {recall} & Precision {precision}")

    name = f"{chosen_detector_name}_{chosen_dataset_name}_"
    title = name + f"Rcl={recall}_Prec={precision}"
    draw_plots(X, outliers, name, title, RESULTS_DIR, save_stats)

    display_finish()


# DEF ------------------------------------------------------------------------ #
def prepare_args() -> Namespace:
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        "-d", "--detector", type=str, choices=DETECTORS.keys(), help="Name of detector"
    )
    arg_parser.add_argument(
        "-ds", "--dataset", type=str, choices=DATASETS.keys(), help="Name of dataset"
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
