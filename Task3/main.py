from argparse import ArgumentParser, Namespace
from typing import Dict, List

import pandas as pd
from sesd import seasonal_esd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from module.reader import read_dataset_1, read_dataset_2, read_dataset_3
from module.utils import create_directory, display_finish, run_main

"""
    How to run:
        python main.py -s -a arima
"""

# VAR ------------------------------------------------------------------------ #
RESULTS_DIR = "results/"
ALGORITHM_NAMES: List[str] = ["arima", "ets", "shesd"]


# MAIN ----------------------------------------------------------------------- #
def main() -> None:
    args = prepare_args()
    chosen_algorithm_name = args.algorithm
    save_stats = args.save
    create_directory(RESULTS_DIR)

    datasets: Dict[str, pd.DataFrame] = {
        "dataset_1": read_dataset_1(),
        "dataset_2": read_dataset_2(),
        "dataset_3": read_dataset_3(),
    }

    for dataset in datasets:
        # TODO SET PROPER PARAMS
        if chosen_algorithm_name == ALGORITHM_NAMES[0]:
            pred = ARIMA(datasets[dataset].to_numpy(), order=(1, 1, 2)).fit().predict()
        elif chosen_algorithm_name == ALGORITHM_NAMES[2]:
            pred = ETSModel(datasets[dataset].to_numpy()).fit().predict()
        elif chosen_algorithm_name == ALGORITHM_NAMES[2]:
            anomalies = seasonal_esd(datasets[dataset].to_numpy())

    display_finish()


# DEF ------------------------------------------------------------------------ #
def prepare_args() -> Namespace:
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        "-a", "--algorithm", type=str, choices=ALGORITHM_NAMES,
        help="Name of algorithm detecting special patterns"
    )
    arg_parser.add_argument(
        "-s", "--save", default=False, action="store_true", help="Save charts to files"
    )

    return arg_parser.parse_args()


# __MAIN__ ------------------------------------------------------------------- #
if __name__ == "__main__":
    run_main(main)
