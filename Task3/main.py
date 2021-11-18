from argparse import ArgumentParser, Namespace
from typing import Dict

import numpy as np

from module.detector.ArimaDetector import ArimaDetector
from module.detector.Detector import Detector
from module.detector.EtsDetector import EtsDetector
from module.detector.ShesdDetector import ShesdDetector
from module.reader import read_air_passengers, read_dataset_2, read_dataset_3
from module.utils import create_directory, display_finish, run_main

"""
    How to run:
        python main.py -s -a arima
"""

# VAR ------------------------------------------------------------------------ #
RESULTS_DIR = "results/"

DETECTORS: Dict[str, Detector] = {
    "arima": ArimaDetector(),
    "ets": EtsDetector(),
    "shesd": ShesdDetector()
}

DATASETS: Dict[str, np.ndarray] = {
    "air_passengers": read_air_passengers(),
    "dataset_2": read_dataset_2(),
    "dataset_3": read_dataset_3(),
}


# MAIN ----------------------------------------------------------------------- #
def main() -> None:
    args = prepare_args()
    chosen_detector_name = args.detector
    save_stats = args.save
    create_directory(RESULTS_DIR)

    detector: Detector = DETECTORS[chosen_detector_name]
    for dataset in DATASETS:
        detector.detect(DATASETS[dataset])

        if save_stats:
            pass

    display_finish()


# DEF ------------------------------------------------------------------------ #
def prepare_args() -> Namespace:
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        "-a", "--detector", type=str, choices=DETECTORS.keys(), help="Name of detector"
    )
    arg_parser.add_argument(
        "-s", "--save", default=False, action="store_true", help="Save charts to files"
    )

    return arg_parser.parse_args()


# __MAIN__ ------------------------------------------------------------------- #
if __name__ == "__main__":
    run_main(main)
