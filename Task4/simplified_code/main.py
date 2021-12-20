import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from pyod.models.abod import ABOD
from pyod.models.lof import LOF
from pyod.models.cof import COF
from reader import read_http_dataset, read_mammography_dataset, read_synthetic_dataset


def find_outliers(dataset, detector, **params):
    X, y = dataset
    d = detector(**params).fit(X)
    y_proba = d.predict_proba(X)[:, 1]
    y_proba[np.isnan(y_proba)] = 0
    return precision_recall_curve(y, y_proba)


def plot_results(synthetic, mammography, http):
    plt.plot(synthetic[0], synthetic[1], label="synthetic")
    plt.plot(mammography[0], mammography[1], label="mammography")
    plt.plot(http[0], http[1], label="http")
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.legend()
    plt.show()


# ABOD experiments
synthetic = find_outliers(read_synthetic_dataset(), ABOD, n_neighbors=50)
mammography = find_outliers(read_mammography_dataset(), ABOD, n_neighbors=50)
http = find_outliers(read_http_dataset(), ABOD, n_neighbors=50)
plot_results(synthetic, mammography, http)

# LOF experiments
synthetic = find_outliers(read_synthetic_dataset(), LOF, n_neighbors=50)
mammography = find_outliers(read_mammography_dataset(), LOF, n_neighbors=50)
http = find_outliers(read_http_dataset(), LOF, n_neighbors=50)
plot_results(synthetic, mammography, http)

# COF experiments
synthetic = find_outliers(read_synthetic_dataset(), COF, n_neighbors=50)
mammography = find_outliers(read_mammography_dataset(), COF, n_neighbors=50)
http = find_outliers(read_http_dataset(), COF, n_neighbors=50)
plot_results(synthetic, mammography, http)
