from argparse import ArgumentParser, Namespace
from typing import Any, Dict, Tuple

import numpy as np

from module.detector.ArimaDetector import ArimaDetector
from module.detector.Detector import Detector
from module.detector.EtsDetector import EtsDetector
from module.detector.ShesdDetector import ShesdDetector
from module.reader import read_air_passengers, read_env_telemetry, read_weather_aus
from module.utils import create_directory, display_finish, run_main

"""
    How to run:
        python main.py -s -ds air_passengers -d shesd -ap 1
"""

# VAR ------------------------------------------------------------------------ #
RESULTS_DIR = "results/"

DETECTORS: Dict[str, Any] = {
    "shesd": ShesdDetector,
    "arima": ArimaDetector,
    "ets": EtsDetector,
}

DATASETS: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
    "air_passengers": read_air_passengers(),
    "env_telemetry": read_env_telemetry(),
    "weather_aus": read_weather_aus(),
}


# MAIN ----------------------------------------------------------------------- #
def main() -> None:
    args = prepare_args()
    chosen_detector_name = args.detector
    chosen_dataset_name = args.dataset
    algorithm_params = args.algorithm_params
    save_stats = args.save
    create_directory(RESULTS_DIR)

    configuration_name = (
        f"{chosen_dataset_name}_{chosen_detector_name}_"
        f"{'_'.join([str(param).replace('.', ',') for param in algorithm_params])}_"
    )
    X, y = DATASETS[chosen_dataset_name]
    detector: Detector = DETECTORS[chosen_detector_name](X, y, configuration_name)
    detector.detect()
    print(detector.calculate_statistics())
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
