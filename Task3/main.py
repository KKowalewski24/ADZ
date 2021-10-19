from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List

import pandas as pd
from sesd import seasonal_esd
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from module.reader import read_dataset_1, read_dataset_2, read_dataset_3
from module.utils import check_if_exists_in_args, check_types_check_style, compile_to_pyc, \
    create_directory, display_finish

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
    chosen_algorithm = args.algorithm
    save_stats = args.save
    create_directory(RESULTS_DIR)

    datasets: Dict[str, pd.DataFrame] = {
        "dataset_1": read_dataset_1(),
        "dataset_2": read_dataset_2(),
        "dataset_3": read_dataset_3(),
    }

    algorithms: Dict[str, Any] = {
        ALGORITHM_NAMES[0]: ARIMA(),
        ALGORITHM_NAMES[1]: ETSModel(),
        ALGORITHM_NAMES[2]: None
    }

    for dataset in datasets:
        if chosen_algorithm == ALGORITHM_NAMES[2]:
            seasonal_esd()
        else:
            algorithms[chosen_algorithm].fit().predict()

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
    if check_if_exists_in_args("-t"):
        check_types_check_style()
    elif check_if_exists_in_args("-b"):
        compile_to_pyc()
    else:
        main()
