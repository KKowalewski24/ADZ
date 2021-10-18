import subprocess
import sys
from argparse import ArgumentParser, Namespace
from typing import List

from module.algorithm_type_resolver import prepare_benchmark_algorithms, resolve_clusterizer_type

"""
    How to run:
        python main.py -s -c 
"""

# VAR ------------------------------------------------------------------------ #
CLUSTERIZER_NAMES: List[str] = ["kmeans", "agglomerative"]


# MAIN ----------------------------------------------------------------------- #
def main() -> None:
    args = prepare_args()
    chosen_clusterizer_name = args.clusterizer
    save_charts = args.save

    clusterizer = resolve_clusterizer_type(CLUSTERIZER_NAMES, chosen_clusterizer_name)
    db_scan, lof = prepare_benchmark_algorithms()

    display_finish()


# DEF ------------------------------------------------------------------------ #
def prepare_args() -> Namespace:
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        "-c", "--clusterizer", type=str, choices=CLUSTERIZER_NAMES,
        help="Name of clusterizer"
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
