import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

import plotting
from datasets import get_toy_dataset


def __perform_gridsearch_on_svc(kernel):
  svc = SVC(kernel=kernel, tol=1e-4)

  param_grid = {
    "C": 10. ** np.arange(-3, 4),
    "gamma": np.logspace(-3, 3, 7)
  }

  clf = GridSearchCV(svc, param_grid, n_jobs=-1)
  clf.fit(X_train, y_train)

  scores = clf.cv_results_
  best_params = clf.best_params_
  test_score = clf.score(X_test, y_test)
  mean_cv_accuracies = clf.cv_results_['mean_test_score']
  best_index = scores['params'].index(best_params)

  print(f'Dataset {idx}, kernel = {svc.kernel}: The best SVC parameters found by GridSearchCV are {best_params}.')
  print(f"Dataset {idx}, kernel = {svc.kernel}: Mean cross validated accuracy: {mean_cv_accuracies[best_index]:.4f}")
  print(f"Dataset {idx}, kernel = {svc.kernel}: Test Score: {test_score:.4f}")

  __plot_data_set_with_decision_boundary(clf, svc)


def __plot_data_set_with_decision_boundary(clf, svc):
  plt.figure()
  plt.title(f'Dataset {idx}, kernel = {svc.kernel} with decision boundary')
  plt.xlabel('x1')
  plt.ylabel('x2')
  plotting.plot_decision_boundary(X_train, clf)
  plotting.plot_dataset(X_train, X_test, y_train, y_test)
  plt.show()


if __name__ == '__main__':
  for idx in [1, 2, 3]:
    X_train, X_test, y_train, y_test = get_toy_dataset(idx)
    __perform_gridsearch_on_svc('linear')
    __perform_gridsearch_on_svc('rbf')
