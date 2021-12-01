from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List, Tuple

import numpy as np

from module.detector.BlockNestedLoopDetector import BlockNestedLoopDetector
from module.detector.Detector import Detector
from module.detector.FastAbodDetector import FastAbodDetector
from module.reader import read_http_dataset, read_mammography_dataset, read_synthetic_dataset
from module.utils import create_directory, display_finish, run_main

"""
    How to run: 
        python main.py -s -ds synthetic -d fast_abod
"""

# VAR ------------------------------------------------------------------------ #
DETECTORS: Dict[str, Any] = {
    "fast_abod": FastAbodDetector,
    "bnl": BlockNestedLoopDetector,
}

DATASETS: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {
    "http": read_http_dataset(),
    "mammography": read_mammography_dataset(),
    "synthetic": read_synthetic_dataset(),
}

EXPERIMENTS: List[Tuple[str, str, List[Dict[str, Any]]]] = [
    ("fast_abod", "http", [
        {"contamination": 0.1, "n_neighbors": 5},
        {"contamination": 0.1, "n_neighbors": 25},
        {"contamination": 0.5, "n_neighbors": 5},
        {"contamination": 0.05, "n_neighbors": 5},
    ]),
    ("fast_abod", "mammography", [
        {"contamination": 0.1, "n_neighbors": 5},
        {"contamination": 0.1, "n_neighbors": 25},
        {"contamination": 0.5, "n_neighbors": 5},
        {"contamination": 0.05, "n_neighbors": 5},
    ]),
    ("fast_abod", "synthetic", [
        {"contamination": 0.1, "n_neighbors": 5},
        {"contamination": 0.1, "n_neighbors": 25},
        {"contamination": 0.5, "n_neighbors": 5},
        {"contamination": 0.05, "n_neighbors": 5},
    ]),

    ("bnl", "http", [{}]),
    ("bnl", "mammography", [{}]),
    ("bnl", "synthetic", [{}]),
]


# MAIN ----------------------------------------------------------------------- #
def main() -> None:
    args = prepare_args()
    chosen_detector_name = args.detector
    chosen_dataset_name = args.dataset
    save_results = args.save
    create_directory(Detector.RESULTS_DIR)

    params_list = [
        experiment[2]
        for experiment in EXPERIMENTS
        if experiment[0] == chosen_detector_name and experiment[1] == chosen_dataset_name
    ][0]

    dataset = DATASETS[chosen_dataset_name]
    for params in params_list:
        configuration_name = (
            f"{chosen_dataset_name}_{chosen_detector_name}_"
            f"{'_'.join([param + '=' + str(params[param]).replace('.', ',') for param in params])}"
        )
        detector = DETECTORS[chosen_detector_name](dataset, configuration_name)
        detector.detect(params)
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
        "-s", "--save", default=False, action="store_true", help="Save results to files"
    )

    return arg_parser.parse_args()


# __MAIN__ ------------------------------------------------------------------- #
if __name__ == "__main__":
    run_main(main)
