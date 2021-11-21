from argparse import ArgumentParser, Namespace
from typing import Dict, Tuple

import numpy as np

from module.detector.AbodDetector import AbodDetector
from module.detector.Detector import Detector
from module.reader import read_http_dataset, read_mammography_dataset, read_synthetic_dataset
from module.utils import create_directory, display_finish, run_main

"""
    How to run: 
        python main.py -s -ds synthetic -d abod -ap 1
"""

# VAR ------------------------------------------------------------------------ #
DETECTORS: Dict[str, Detector] = {
    "abod": AbodDetector(),
}

DATASETS: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {
    "http": read_http_dataset(),
    "mammography": read_mammography_dataset(),
    "synthetic": read_synthetic_dataset(),
}


# MAIN ----------------------------------------------------------------------- #
def main() -> None:
    args = prepare_args()
    chosen_detector_name = args.detector
    chosen_dataset_name = args.dataset
    algorithm_params = args.algorithm_params
    save_results = args.save
    create_directory(Detector.RESULTS_DIR)

    detector = DETECTORS[chosen_detector_name]
    detector.detect(DATASETS[chosen_dataset_name])
    detector.show_results(save_results)

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
        "-s", "--save", default=False, action="store_true", help="Save results to files"
    )

    return arg_parser.parse_args()


# __MAIN__ ------------------------------------------------------------------- #
if __name__ == "__main__":
    run_main(main)
