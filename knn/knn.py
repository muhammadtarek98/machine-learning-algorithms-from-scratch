import numpy as np
from collections import Counter

np.random.seed(42)
class Knn:
    def __init__(self, k=3):
        """
        k: num of nearest neighbours
        """
        self.k = k

    @staticmethod
    def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y

    @staticmethod
    def accuracy(y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        distances = [Knn.euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_idx = np.argsort(distances)[: self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.Y_train[i] for i in k_idx]
        # return the most common class label
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]
