from argparse import ArgumentParser, Namespace
from typing import Dict, Tuple, List, Any

import numpy as np

from module.detector.AbodDetector import AbodDetector
from module.detector.Detector import Detector
from module.reader import read_http_dataset, read_mammography_dataset, read_synthetic_dataset
from module.utils import create_directory, display_finish, run_main

"""
    How to run: 
        python main.py -s -ds synthetic -d abod
"""

# VAR ------------------------------------------------------------------------ #
DETECTORS: Dict[str, Any] = {
    "abod": AbodDetector,
}

DATASETS: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {
    "http": read_http_dataset(),
    "mammography": read_mammography_dataset(),
    "synthetic": read_synthetic_dataset(),
}

EXPERIMENTS: List[Tuple[str, str, List[Dict[str, Any]]]] = [
    ("abod", "http", [{}]),
    ("abod", "mammography", [{}]),
    ("abod", "synthetic", [{}]),
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
            f"{'_'.join([str(param).replace('.', ',') for param in params])}"
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
