import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator

import plotting
from datasets import get_toy_dataset


def loss(w, b, C, X, y):
  return 1/2 * w.T @ w + C * np.sum(np.clip(1 - y * (X @ w + b), a_min=0, a_max=None))


def grad(w, b, C, X, y):
  # define condition for piece-wise gradient
  grad_condition = 1 - y * (X @ w + b)  # piece-wise condition for gradient being >= 0
  X_grad = np.where(np.repeat(grad_condition.reshape((X.shape[0], 1)) >= 0, 2, axis=1), X, 0)

  # calculate grad for w
  inner_exp = -X_grad * y.reshape((X.shape[0], 1))
  grad_w = w + C * np.sum(inner_exp, axis=0)

  # calculate grad for b
  y_grad = np.where(grad_condition >= 0, y, 0)
  grad_b = C * np.sum(-y_grad)

  return grad_w, grad_b


class LinearSVM(BaseEstimator):

  def __init__(self, C=1, eta=1e-3, max_iter=1000):
    self.C = C
    self.max_iter = max_iter
    self.eta = eta
    self.tolerance = self.eta

  def fit(self, X, y):
    # TODO: initialize w and b. Does the initialization matter?
    # convert y: {0,1} -> -1, 1
    y = np.where(y == 0, -1, 1)
    self.w = np.random.normal(loc=0, scale=1.0, size=X.shape[1])
    self.b = 0.
    # self.w = [6.87296361, 9.56862135]
    # self.b = -10
    loss_list = []

    for j in range(self.max_iter):
      grad_w, grad_b = grad(self.w, self.b, self.C, X, y)
      self.w = self.w - self.eta * grad_w
      self.b = self.b - self.eta * grad_b
      loss_list.append(loss(self.w, self.b, self.C, X, y))

      if len(loss_list) >= 2 and np.isclose(loss_list[-2], loss_list[-1], rtol=self.tolerance):
        break

    # print(f'Model fit: C = {self.C}, eta = {self.eta}, tolerance = {self.tolerance}, num_epochs = {len(loss_list)}')

    return loss_list

  def predict(self, X):
    y_regressed = X @ self.w + self.b  # predition regression value of model
    y_pred = np.where(np.tanh(y_regressed) > 0, 1, -1)  # decide on binary class label based on regressed value

    # converting y_pred from {-1, 1} to {0, 1}
    return np.where(y_pred == -1, 0, 1)

  def score(self, X, y):
    y_pred = self.predict(X)
    return np.mean(y_pred == y)
