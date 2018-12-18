#!/usr/local/bin/python3
# Brandon Tarney
# SRC - https://cerkauskas.com/blog/k-nearest-neighbor/
# SRC - https://cerkauskas.com/blog/faster-k-nearest-neigbor-numpy/

#BASIC STUFF:
def train(self, data, labels):
    self._training_data = data
    self._training_labels = labels

from math import sqrt
def _distance(self, a, b):
    total = 0
    
    for i in range(len(a)):
        total += (a[i] - b[i])**2
    
    return sqrt(total)


#NUMPY OPTIMIZED AWESOME:
class KNearestNeighbor(object):
    # Prediction option 1....
    def predict(self, features):
        target = np.repeat([features], len(self._training_data), axis=0)
        dists = np.sum(np.square(self._training_data - target), axis=1)
        indices = dists.argsort()[:self._k]
        labels = map(lambda x: {'label': self._training_labels[x]}, indices)

        return self._choose_majority(labels)

    #Prediction option 2 ....
    def predict(self, features):
        # ....
        # We don't actually need to store labels in dictionary
        labels = map(lambda x: self._training_labels[x], indices)

        return np.bincount(labels).argmax()
