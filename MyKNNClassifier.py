from numpy.linalg import norm as euclidean_distance
import cv2
import imutils
from collections import defaultdict


class MyKNNClassifier(object):
    def __init__(self,n_neighbors=5):
        self.n_neighbors=n_neighbors

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def _distance(self, data1, data2):
        """returns Euclidian distance"""
        return euclidean_distance(data1 - data2)

    def _compute_weights(self, distances):

        return [(1, y) for d, y in distances]

    def _predict_one(self, test):
        distances = sorted((self._distance(x, test), y) for x, y in zip(self.X, self.y))
        neighbors = distances[:self.n_neighbors]
        weights = self._compute_weights(neighbors)
        weights_by_class = defaultdict(list)
        for d, c in weights:
            weights_by_class[c].append(d)
        return max((sum(val), key) for key, val in weights_by_class.items())[1]


    def predict(self, X):
        return [self._predict_one(i) for i in X]
