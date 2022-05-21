import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV

import plotting
from datasets import get_toy_dataset
from task1_1 import KNearestNeighborsClassifier
from sklearn.model_selection import cross_val_score


def __determine_best_value_for_k_using_grid_search():
  """ Determine best value for k using 5-fold cross-validation using grid-search
  """

  knn = KNearestNeighborsClassifier()

  param_grid = {
    "k": range(1, 101)
  }

  clf = GridSearchCV(knn, param_grid, n_jobs=-1, return_train_score=True)
  clf.fit(X_train, y_train)
  print(f"Determine best value for k using 5-fold cross-validation using grid-search: k = {clf.best_params_['k']}, accuracy_score = {clf.best_score_:.4f}")

  # plot training and validation score for all values of k
  plt.figure()
  plt.title(f'Noisy dataset 2: training and validation scores')
  plt.xlabel('k')
  plt.ylabel('accuracy')
  plt.plot(clf.cv_results_['mean_train_score'], label='mean training scores')
  plt.plot(clf.cv_results_['mean_test_score'], label='mean validation scores')
  plt.legend()
  plt.show()

  return clf.best_params_['k']


def __plot_decision_boundaries():
  for k in [1, 5, 20, 50, 100]:
    clf = KNearestNeighborsClassifier(k)
    clf.fit(X_train, y_train)

    cv_score = cross_val_score(clf, X_test, y_test, n_jobs=-1)
    print(f"Mean cross validation score for k={k}: {np.mean(cv_score):.4f}")

    test_score = clf.score(X_test, y_test)
    print(f"Test Score for k={k}: {test_score:.4f}")

    # plot dataset with decision boundary
    plt.figure()
    plt.title(f'Noisy dataset 2 for k = {k} with decision boundary')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plotting.plot_decision_boundary(X_train, clf)
    plotting.plot_dataset(X_train, X_test, y_train, y_test)
    plt.show()


def __report_performance_for_best_k(k):
  """ Report the performance of the classifier with your chosen value of k on the test set (we choose k=15 since it performed best on the validation data)
  """

  clf = KNearestNeighborsClassifier(k)
  clf.fit(X_train, y_train)
  test_score = clf.score(X_test, y_test)
  print(f"Performance of classifier on test set using best validation score for k={k}: test accuracy = {test_score:.4f}")


if __name__ == '__main__':
  X_train, X_test, y_train, y_test = get_toy_dataset(2, apply_noise=True)

  __plot_decision_boundaries()
  best_k = __determine_best_value_for_k_using_grid_search()
  __report_performance_for_best_k(best_k)
