import subprocess
import sys
from argparse import ArgumentParser, Namespace
from typing import List

from skmultiflow.data import FileStream

from module.algorithm_type_resolver import resolve_classifier_type, resolve_detector_type
from module.classifier import classify
from module.data_generator import generate_data
from module.plot import draw_plots
from module.utils import create_directory

"""
    How to run:
        Generating dataset: python main.py -ds
        Running detection:  python main.py -d ddm -c knn -s
"""

# VAR ------------------------------------------------------------------------ #
DATASET_DIR: str = "data/"
DATASET_PATH: str = DATASET_DIR + "data.csv"
DATASET_ROWS_NUMBER: int = 20000

DETECTOR_NAMES: List[str] = ["ddm", "eddm", "adwin", "kswin", "ph"]
CLASSIFIER_NAMES: List[str] = ["knn", "vfdt"]
TRAIN_SIZE: int = 5000


# MAIN ----------------------------------------------------------------------- #
def main() -> None:
    args = prepare_args()
    generate_dataset = args.dataset
    chosen_classifier = args.classifier
    chosen_detector = args.detector
    save_charts = args.save
    create_directory(DATASET_DIR)

    if generate_dataset:
        generate_data(DATASET_PATH, DATASET_ROWS_NUMBER)
        return

    dataset: FileStream = FileStream(DATASET_PATH)
    detector = resolve_detector_type(DETECTOR_NAMES, chosen_detector)
    classifier = resolve_classifier_type(CLASSIFIER_NAMES, chosen_classifier)
    changes, warnings, accuracy_trend, train_size_range = classify(
        detector, classifier, dataset, TRAIN_SIZE
    )
    draw_plots(
        changes, warnings, accuracy_trend, train_size_range,
        chosen_detector, chosen_classifier, save_charts
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


# UTIL ----------------------------------------------------------------------- #
def check_types_check_style() -> None:
    subprocess.call(["mypy", "."])
    subprocess.call(["flake8", "."])


def compile_to_pyc() -> None:
    subprocess.call(["python", "-m", "compileall", "."])


def check_if_exists_in_args(arg: str) -> bool:
    return arg in sys.argv


def display_finish() -> None:
    print("------------------------------------------------------------------------")
    print("FINISHED")
    print("------------------------------------------------------------------------")


# __MAIN__ ------------------------------------------------------------------- #
if __name__ == "__main__":
    if check_if_exists_in_args("-t"):
        check_types_check_style()
    elif check_if_exists_in_args("-b"):
        compile_to_pyc()
    else:
        main()
