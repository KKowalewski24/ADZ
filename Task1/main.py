import subprocess
import sys
from argparse import ArgumentParser, Namespace
from typing import List

from module.algorithm_type_resolver import resolve_classifier_type, resolve_detector_type
from module.classifier import classify
from module.plot import draw_plots

"""
    How to run:
        python main.py -d ddm -c knn 
"""

# VAR ------------------------------------------------------------------------ #
DETECTOR_NAMES: List[str] = ["ddm", "eddm", "adwin", "kswin"]
CLASSIFIER_NAMES: List[str] = ["knn", "vfdt"]


# MAIN ----------------------------------------------------------------------- #
def main() -> None:
    args = prepare_args()
    chosen_classifier = args.classifier
    chosen_detector = args.detector
    save_charts = args.save

    detector = resolve_detector_type(DETECTOR_NAMES, chosen_detector)
    classifier = resolve_classifier_type(CLASSIFIER_NAMES, chosen_classifier)
    classify(detector, classifier)
    # draw_plots()

    display_finish()


# DEF ------------------------------------------------------------------------ #
def prepare_args() -> Namespace:
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        "-d", "--detector", required=True, type=str, choices=DETECTOR_NAMES,
        help="Name of detector"
    )
    arg_parser.add_argument(
        "-c", "--classifier", required=True, type=str, choices=CLASSIFIER_NAMES,
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
