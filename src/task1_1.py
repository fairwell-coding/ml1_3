import numpy as np
from sklearn.base import BaseEstimator


class KNearestNeighborsClassifier(BaseEstimator):
  def __init__(self, k=1):
    self.k = k

  def fit(self, X, y):
    self.X_train = X
    self.y_train = y

    return self

  def score(self, X, y):
    y_pred = self.predict(X)
    return np.mean(y_pred == y)

  def predict(self, X):
    predictions = []

    for i in range(X.shape[0]):
      eucledian_distances = np.linalg.norm(self.X_train - X[i].reshape(-1, X[i].shape[0]), axis=1)
      sorted_indices = np.argsort(eucledian_distances)[:self.k]  # first k indices of eucledian distances between all training samples and current test sample in ascending order
      closest_classes = self.y_train[sorted_indices]

      num_class_1 = np.count_nonzero(closest_classes)
      num_class_0 = closest_classes.shape[0] - num_class_1
      predictions.append(np.argmax([num_class_0, num_class_1]))

    return predictions
