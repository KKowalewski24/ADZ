from argparse import ArgumentParser, Namespace
from typing import Dict, Tuple

import pandas as pd

from module.reader import read_dataset_1, read_dataset_2, read_dataset_3
from module.utils import create_directory, display_finish, run_main

"""
    How to run: python main.py -s -a 
        
"""

# VAR ------------------------------------------------------------------------ #
RESULTS_DIR = "results/"

ALGORITHM_SETUP: Dict[str, Tuple] = {

}


# MAIN ----------------------------------------------------------------------- #
def main() -> None:
    args = prepare_args()
    chosen_algorithm_name = args.algorithm
    save_results = args.save
    create_directory(RESULTS_DIR)

    print("Reading datasets ...")
    datasets: Dict[str, pd.DataFrame] = {
        "dataset_1": read_dataset_1(),
        "dataset_2": read_dataset_2(),
        "dataset_3": read_dataset_3(),
    }

    for dataset in datasets:
        print(f"Detecting dataset: {dataset} ...")

        if save_results:
            pass

    display_finish()


# DEF ------------------------------------------------------------------------ #
def prepare_args() -> Namespace:
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        "-a", "--algorithm", type=str, choices=ALGORITHM_SETUP.keys(), help="Name of algorithm"
    )
    arg_parser.add_argument(
        "-s", "--save", default=False, action="store_true", help="Save results to files"
    )

    return arg_parser.parse_args()


# __MAIN__ ------------------------------------------------------------------- #
if __name__ == "__main__":
    run_main(main)
