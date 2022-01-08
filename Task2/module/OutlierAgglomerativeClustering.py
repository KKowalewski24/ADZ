import numpy as np
from sklearn.cluster import AgglomerativeClustering


class OutlierAgglomerativeClustering(AgglomerativeClustering):

    def __init__(self, distance_threshold, outlier_fraction_threshold):
        super().__init__(n_clusters=None,
                         distance_threshold=distance_threshold)
        self.outlier_fraction_threshold = outlier_fraction_threshold


    def fit(self, X, y=None):
        super().fit(X, y)
        for label in np.unique(self.labels_):
            quantity = np.count_nonzero(self.labels_ == label)
            if quantity < self.outlier_fraction_threshold * len(X):
                self.labels_[self.labels_ == label] = -1
        return self
