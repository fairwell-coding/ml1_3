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

  def predict_old(self, X_test):
    """ Implements predict method by looping over test samples.
    """

    predictions = []

    for i in range(X_test.shape[0]):
      eucledian_distances = np.linalg.norm(self.X_train - X_test[i].reshape(-1, X_test[i].shape[0]), axis=1)
      sorted_indices = np.argsort(eucledian_distances)[:self.k]  # first k indices of eucledian distances between all training samples and current test sample in ascending order
      closest_classes = self.y_train[sorted_indices]

      num_class_1 = np.count_nonzero(closest_classes)
      num_class_0 = closest_classes.shape[0] - num_class_1
      predictions.append(np.argmax([num_class_0, num_class_1]))

    return predictions

  def predict(self, X_test):
    """ Implements predict by vectorization of test samples.
    """

    # eucledian distance: train x test x coordinates
    X_train_reshaped = self.X_train.reshape((self.X_train.shape[0], -1, self.X_train.shape[1]))
    X_test_reshaped = X_test.reshape((-1, X_test.shape[0], X_test.shape[1]))
    eucledian_distances = np.linalg.norm(X_train_reshaped - X_test_reshaped, axis=2)

    # sort ascending: k x test
    sorted_indices = np.argsort(eucledian_distances, axis=0)[:self.k]

    # closest classes: test x k
    closest_classes = []
    for i in range(X_test.shape[0]):
      closest_classes.append(self.y_train[sorted_indices[:, i]])

    # predictions: test x 1
    num_class_1 = np.count_nonzero(closest_classes, axis=1)
    num_class_0 = self.k - num_class_1
    predictions = np.argmax([num_class_0, num_class_1], axis=0)

    return predictions
