import os
from argparse import ArgumentParser, Namespace
from typing import Dict

from skmultiflow.data import FileStream
from skmultiflow.drift_detection import ADWIN, DDM, HDDM_A, KSWIN, PageHinkley

from module.classifier import classify
from module.plot import draw_plots
from module.preprocessing import preprocess_data
from module.utils import display_finish, run_main

"""
    How to run:
        Running detection:  python main.py -d ddm -ws 3000 -s
"""

# VAR ------------------------------------------------------------------------ #
DATASET_DIR: str = "data/"
ORIGINAL_DATASET_PATH: str = DATASET_DIR + "weatherAUS.csv"
DATASET_PATH: str = DATASET_DIR + "filtered_weatherAUS.csv"

# TODO SET PARAMS
DETECTORS_SETUP: Dict = {
    "adwin": (ADWIN, []),
    "ddm": (DDM, []),
    "hddm_a": (HDDM_A, []),
    "kswin": (KSWIN, []),
    "ph": (PageHinkley, [])
}


# MAIN ----------------------------------------------------------------------- #
def main() -> None:
    args = prepare_args()
    chosen_detector_name = args.detector
    window_size = args.window_size
    save_charts = args.save

    if not os.path.exists(DATASET_PATH):
        print("Processing data...")
        preprocess_data(ORIGINAL_DATASET_PATH, DATASET_PATH)

    print("Classifying...")
    changes, warnings, accuracy_trend, window_size_range = classify(
        DETECTORS_SETUP[chosen_detector_name][0](*DETECTORS_SETUP[chosen_detector_name][1]),
        FileStream(DATASET_PATH), window_size
    )

    print("Drawing plots...")
    draw_plots(
        changes, warnings, accuracy_trend, window_size_range,
        chosen_detector_name, save_charts
    )

    display_finish()


# DEF ------------------------------------------------------------------------ #
def prepare_args() -> Namespace:
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        "-d", "--detector", required=True, type=str, choices=DETECTORS_SETUP.keys(),
        help="Name of detector"
    )
    arg_parser.add_argument(
        "-ws", "--window_size", required=True, type=int, help="Window size"
    )
    arg_parser.add_argument(
        "-s", "--save", default=False, action="store_true", help="Save charts to files"
    )

    return arg_parser.parse_args()


# __MAIN__ ------------------------------------------------------------------- #
if __name__ == "__main__":
    run_main(main)
