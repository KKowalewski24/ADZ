import numpy as np
from sklearn.cluster import KMeans


class OutlierKMeans(KMeans):

    def __init__(self,
                 n_clusters,
                 outlier_fraction_threshold,
                 *,
                 init="k-means++",
                 n_init=10,
                 max_iter=300,
                 tol=1e-4,
                 verbose=0,
                 random_state=None,
                 copy_x=True,
                 algorithm="auto"):
        super().__init__(n_clusters,
                         init=init,
                         n_init=n_init,
                         max_iter=max_iter,
                         tol=tol,
                         verbose=verbose,
                         random_state=random_state,
                         copy_x=copy_x,
                         algorithm=algorithm)
        self.outlier_fraction_threshold = outlier_fraction_threshold


    def fit(self, X, y=None, sample_weight=None):
        super().fit(X, y, sample_weight)
        for label in np.unique(self.labels_):
            quantity = np.count_nonzero(self.labels_ == label)
            if quantity < self.outlier_fraction_threshold * len(X):
                self.labels_[self.labels_ == label] = -1
        return self
