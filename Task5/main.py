from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List, Tuple

import numpy as np

from module.detector.Detector import Detector
from module.reader import read_synthetic_dataset
from module.utils import create_directory, display_finish, run_main

"""
    How to run: 
        python main.py -s -d kmeans -ds synthetic
"""

# VAR ------------------------------------------------------------------------ #

DETECTORS: Dict[str, Any] = {
    "kmeans": 0
}

DATASETS: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
    "synthetic": read_synthetic_dataset(),
}

EXPERIMENTS: List[Tuple[str, str, List[Dict[str, Any]]]] = [
    ("kmeans", "synthetic", [{}]),
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
        statistics = detector.calculate_statistics()
        detector.show_results(save_results)

        summary = (f"{chosen_detector_name} & {chosen_dataset_name} & "
                   + " & ".join([str(statistics[stat]) for stat in statistics]))
        print(summary)
        with open("summary.txt", "a") as file:
            file.write(summary + "\n")

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
