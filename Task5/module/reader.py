from typing import Tuple

import numpy as np

DATASET_DIR: str = "data/"
SYNTHETIC_DATASET_PATH = f"{DATASET_DIR}synthetic_dataset.csv"


def read_synthetic_dataset() -> Tuple[np.ndarray, np.ndarray]:
    # TODO
    # if not os.path.exists(SYNTHETIC_DATASET_PATH):
    #     pass

    # data = pd.read_csv(SYNTHETIC_DATASET_PATH)
    pass


def read_http_dataset() -> Tuple[np.ndarray, np.ndarray]:
    # TODO
    pass


def read_mammography_dataset() -> Tuple[np.ndarray, np.ndarray]:
    # TODO
    pass
