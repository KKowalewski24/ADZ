from argparse import ArgumentParser, Namespace
from concurrent.futures.process import ProcessPoolExecutor
from typing import Any, Dict, List, Tuple

import numpy as np

from module.detector.Detector import Detector
from module.detector.FuzzyCMeansDetector import FuzzyCMeansDetector
from module.detector.KMeansDetector import KMeansDetector
from module.reader import read_http_dataset, read_mammography_dataset, read_synthetic_dataset
from module.utils import create_directory, display_finish, run_main

"""
    How to run: 
        python main.py -s -d cmeans -ds synthetic
"""

# VAR ------------------------------------------------------------------------ #

DETECTORS: Dict[str, Any] = {
    "cmeans": FuzzyCMeansDetector,
    "kmeans": KMeansDetector
}

DATASETS: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
    "synthetic": read_synthetic_dataset(),
    "mammography": read_mammography_dataset(),
    "http": read_http_dataset(),
}

SYNTHETIC_SETUP: List[Dict[str, Any]] = [
    {
        "n_clusters": 2,
        "outlier_fraction_threshold": 0.01,
    },
    {
        "n_clusters": 2,
        "outlier_fraction_threshold": 0.1,
    },
    {
        "n_clusters": 2,
        "outlier_fraction_threshold": 0.2,
    },
    {
        "n_clusters": 5,
        "outlier_fraction_threshold": 0.01,
    },
    {
        "n_clusters": 5,
        "outlier_fraction_threshold": 0.1,
    },
    {
        "n_clusters": 5,
        "outlier_fraction_threshold": 0.2,
    },
    {
        "n_clusters": 10,
        "outlier_fraction_threshold": 0.01,
    },
    {
        "n_clusters": 10,
        "outlier_fraction_threshold": 0.1,
    },
    {
        "n_clusters": 10,
        "outlier_fraction_threshold": 0.2,
    },
    {
        "n_clusters": 15,
        "outlier_fraction_threshold": 0.01,
    },
    {
        "n_clusters": 15,
        "outlier_fraction_threshold": 0.1,
    },
    {
        "n_clusters": 15,
        "outlier_fraction_threshold": 0.2,
    },
    {
        "n_clusters": 20,
        "outlier_fraction_threshold": 0.01,
    },
    {
        "n_clusters": 20,
        "outlier_fraction_threshold": 0.1,
    },
    {
        "n_clusters": 20,
        "outlier_fraction_threshold": 0.2,
    },
]

MAMMOGRAPHY_SETUP: List[Dict[str, Any]] = [
    {
        "n_clusters": 40,
        "outlier_fraction_threshold": 0.005,
    },
    {
        "n_clusters": 50,
        "outlier_fraction_threshold": 0.005,
    }
]

HTTP_SETUP: List[Dict[str, Any]] = [
    {
        "n_clusters": 5,
        "outlier_fraction_threshold": 0.1,
    },
    {
        "n_clusters": 5,
        "outlier_fraction_threshold": 0.2,
    },
    {
        "n_clusters": 8,
        "outlier_fraction_threshold": 0.005,
    },
    {
        "n_clusters": 8,
        "outlier_fraction_threshold": 0.01,
    },
    {
        "n_clusters": 8,
        "outlier_fraction_threshold": 0.05,
    },
    {
        "n_clusters": 10,
        "outlier_fraction_threshold": 0.005,
    },
    {
        "n_clusters": 10,
        "outlier_fraction_threshold": 0.01,
    },
    {
        "n_clusters": 10,
        "outlier_fraction_threshold": 0.05,
    },
    {
        "n_clusters": 10,
        "outlier_fraction_threshold": 0.1,
    },
    {
        "n_clusters": 10,
        "outlier_fraction_threshold": 0.2,
    },
    {
        "n_clusters": 15,
        "outlier_fraction_threshold": 0.05,
    },
    {
        "n_clusters": 15,
        "outlier_fraction_threshold": 0.1,
    },
    {
        "n_clusters": 20,
        "outlier_fraction_threshold": 0.01,
    },
    {
        "n_clusters": 20,
        "outlier_fraction_threshold": 0.05,
    },
]

EXPERIMENTS: List[Tuple[str, str, List[Dict[str, Any]]]] = [
    ("cmeans", "synthetic", SYNTHETIC_SETUP),
    ("kmeans", "synthetic", SYNTHETIC_SETUP),

    ("cmeans", "mammography", MAMMOGRAPHY_SETUP),
    ("kmeans", "mammography", MAMMOGRAPHY_SETUP),

    ("cmeans", "http", HTTP_SETUP),
    ("kmeans", "http", HTTP_SETUP)
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
    params_list_len = len(params_list)
    with ProcessPoolExecutor() as executor:
        executor.map(
            run_parallel,
            params_list,
            [dataset] * params_list_len,
            [chosen_dataset_name] * params_list_len,
            [chosen_detector_name] * params_list_len,
            [save_results] * params_list_len
        )

    display_finish()


def run_parallel(
        params: Dict[str, Any], dataset: Tuple[np.ndarray, np.ndarray],
        chosen_dataset_name: str, chosen_detector_name: str, save_results: bool
) -> Any:
    configuration_name = (
        f"{chosen_dataset_name}_{chosen_detector_name}_"
        f"{'_'.join([param + '=' + str(params[param]).replace('.', ',') for param in params])}"
    )
    detector = DETECTORS[chosen_detector_name](dataset, configuration_name)
    detector.detect(params)
    statistics = detector.calculate_statistics()
    detector.show_results(save_results)

    summary = (
            f"{chosen_detector_name} & {chosen_dataset_name} & "
            + " & ".join([str(params[param]) for param in params]) + " & "
            + " & ".join([str(statistics[stat]) for stat in statistics])
    )
    print(summary)
    with open("summary.txt", "a") as file:
        file.write(summary + "\n")


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
