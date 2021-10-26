import os
from argparse import ArgumentParser, Namespace
from typing import List

from skmultiflow.data import FileStream

from module.algorithm_type_resolver import resolve_classifier_type, resolve_detector_type
from module.classifier import classify
from module.data_generator import generate_data, preprocess_data
from module.plot import draw_plots
from module.utils import create_directory, display_finish, run_main

"""
    How to run:
        Generating dataset: python main.py -ds
        Running detection:  python main.py -d ddm -c knn -s
"""

# VAR ------------------------------------------------------------------------ #
GENERATED_DATASET_ROWS_NUMBER: int = 20000
DATASET_DIR: str = "data/"
ORIGINAL_DATASET_PATH: str = DATASET_DIR + "weatherAUS.csv"
DATASET_PATH: str = DATASET_DIR + "filtered_weatherAUS.csv"

DETECTOR_NAMES: List[str] = ["adwin", "ddm", "hddm_a", "kswin", "ph"]
CLASSIFIER_NAMES: List[str] = ["knn", "vfdt"]
TRAIN_SIZE: int = 3000


# MAIN ----------------------------------------------------------------------- #
def main() -> None:
    args = prepare_args()
    generate_dataset = args.dataset
    chosen_classifier_name = args.classifier
    chosen_detector_name = args.detector
    save_charts = args.save

    if generate_dataset:
        create_directory(DATASET_DIR)
        generate_data(DATASET_PATH, GENERATED_DATASET_ROWS_NUMBER)
        return

    if not os.path.exists(DATASET_PATH):
        preprocess_data(ORIGINAL_DATASET_PATH, DATASET_PATH)

    dataset: FileStream = FileStream(DATASET_PATH)
    detector = resolve_detector_type(DETECTOR_NAMES, chosen_detector_name)
    classifier = resolve_classifier_type(CLASSIFIER_NAMES, chosen_classifier_name)
    changes, warnings, accuracy_trend, train_size_range = classify(
        detector, classifier, dataset, TRAIN_SIZE
    )
    draw_plots(
        changes, warnings, accuracy_trend, train_size_range,
        chosen_detector_name, chosen_classifier_name, save_charts
    )

    display_finish()


# DEF ------------------------------------------------------------------------ #
def prepare_args() -> Namespace:
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        "-ds", "--dataset", default=False, action="store_true", help="Generate dataset"
    )
    arg_parser.add_argument(
        "-d", "--detector", type=str, choices=DETECTOR_NAMES,
        help="Name of detector"
    )
    arg_parser.add_argument(
        "-c", "--classifier", type=str, choices=CLASSIFIER_NAMES,
        help="Name of classifier"
    )
    arg_parser.add_argument(
        "-s", "--save", default=False, action="store_true", help="Save charts to files"
    )

    return arg_parser.parse_args()


# __MAIN__ ------------------------------------------------------------------- #
if __name__ == "__main__":
    run_main(main)
