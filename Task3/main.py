from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from module.detector.Detector import Detector
from module.detector.EtsDetector import EtsDetector
from module.detector.ShesdDetector import ShesdDetector
from module.reader import read_air_passengers, read_alcohol_sales, read_gold_price
from module.utils import create_directory, display_finish, run_main

"""
    How to run:
        python main.py -s -d shesd -ds air_passengers
"""

# VAR ------------------------------------------------------------------------ #
RESULTS_DIR = "results/"

DETECTORS: Dict[str, Any] = {
    "shesd": ShesdDetector,
    "ets": EtsDetector,
}

DATASETS: Dict[str, Tuple[pd.DataFrame, np.ndarray]] = {
    "air_passengers": read_air_passengers(),
    "alcohol_sales": read_alcohol_sales(),
    "gold_price": read_gold_price(),
}

EXPERIMENTS: List[Tuple[str, str, List[Dict[str, Any]]]] = [
    ("shesd", "air_passengers", [{}]),
    ("shesd", "alcohol_sales", [{}]),
    ("shesd", "gold_price", [{}]),

    ("ets", "air_passengers", [{}]),
    ("ets", "alcohol_sales", [{}]),
    ("ets", "gold_price", [{}]),
]


# MAIN ----------------------------------------------------------------------- #
def main() -> None:
    args = prepare_args()
    chosen_detector_name = args.detector
    chosen_dataset_name = args.dataset
    save_stats = args.save
    create_directory(RESULTS_DIR)

    params_list = [
        experiment[2]
        for experiment in EXPERIMENTS
        if experiment[0] == chosen_detector_name and experiment[1] == chosen_dataset_name
    ][0]

    for params in params_list:
        configuration_name = (
            f"{chosen_dataset_name}_{chosen_detector_name}_"
            f"{'_'.join([param + '=' + str(params[param]).replace('.', ',') for param in params])}"
        )
        X, y = DATASETS[chosen_dataset_name]
        detector: Detector = DETECTORS[chosen_detector_name](X, y, configuration_name)
        detector.detect(params)
        statistics = detector.calculate_statistics()
        print(" ".join([stat + " & " + str(statistics[stat]) for stat in statistics]))
        detector.show_results(RESULTS_DIR, save_stats)

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
        "-s", "--save", default=False, action="store_true", help="Save charts to files"
    )

    return arg_parser.parse_args()


# __MAIN__ ------------------------------------------------------------------- #
if __name__ == "__main__":
    run_main(main)
