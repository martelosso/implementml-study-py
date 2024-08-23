import numpy as np
from collections import Counter

# Calculate Euclidean distance between two points
def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2)**2))
    return distance

# K-Nearest Neighbors classifier
class KNN:
    def __init__(self, k=3):
        # Number of neighbors to consider
        self.k = k

    # Store training data
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # Predict labels new data points
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    # Predict label for a single data point
    def _predict(self, x):
        # Compute distances to all data points
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # Get closest k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Majority vote
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
        