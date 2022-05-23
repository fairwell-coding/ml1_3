import matplotlib.pyplot as plt
import numpy as np

import plotting
from datasets import get_toy_dataset
from task2_1 import LinearSVM
from sklearn.model_selection import GridSearchCV


RANDOM_SEED = 28


def __find_optimal_parameters_using_gridsearch():
  """ Use gridsearch to find optimal parameters for C and eta.
  """

  param_grid = {
    "C": 10. ** np.arange(-5, 6),
    "eta": np.logspace(-1, -6, 6)
  }
  base_estimator = LinearSVM()
  grid_search = GridSearchCV(base_estimator, param_grid, n_jobs=-1)
  grid_search.fit(X_train, y_train)

  scores = grid_search.cv_results_
  best_params = grid_search.best_params_
  best_score = grid_search.best_score_

  print(f'The best SVM score is {best_score}.')
  print(f'The best SVM parameters found by GridSearchCV are {best_params}.')

  return best_params["C"], best_params["eta"]


def __use_best_parameters_to_instantiate_linear_svm(C, eta):
  """ Use best parameters found by GridSearch to instantiate a LinearSVM.
  """

  svm = LinearSVM(C=C, eta=eta)
  scores = svm.fit(X_train, y_train)

  plt.figure()
  plt.title(f"Training loss curve: C={C}, eta={eta}")
  plt.xlabel('epochs/iterations')
  plt.ylabel('training loss')
  plt.plot(scores)

  plt.figure()
  plt.title(f"Training loss curve (y-axis clipped): C={C}, eta={eta}")
  plt.xlabel('epochs/iterations')
  plt.ylabel('training loss')
  plt.ylim(0, 200)
  plt.plot(scores)

  test_score = svm.score(X_test, y_test)
  print(f"Test Score: {test_score}")

  return svm


def __plot_dataset_with_decision_boundary(svm):
  plt.figure()
  plt.title(f'Dataset 1 (no outlier) with decision boundary C={svm.C}, eta={svm.eta}')
  plt.xlabel('x1')
  plt.ylabel('x2')
  plotting.plot_decision_boundary(X_train, svm)
  plotting.plot_dataset(X_train, X_test, y_train, y_test)
  plt.show()


if __name__ == '__main__':
  np.random.seed(RANDOM_SEED)
  X_train, X_test, y_train, y_test = get_toy_dataset(1, remove_outlier=True)

  # Add additional outliers to test behaviour
  # X_train = np.concatenate((X_train, np.array([-0.5, 5]).reshape((1, 2))), axis=0)
  # X_train = np.concatenate((X_train, np.array([-0.5, 5.2]).reshape((1, 2))), axis=0)
  # y_train = np.hstack((y_train, 0))
  # y_train = np.hstack((y_train, 0))

  C, eta = __find_optimal_parameters_using_gridsearch()
  svm = __use_best_parameters_to_instantiate_linear_svm(100, 1e-3)
  __plot_dataset_with_decision_boundary(svm)
