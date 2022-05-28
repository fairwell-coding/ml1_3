import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

import plotting
from datasets import get_toy_dataset


RANDOM_SEED = 42


def __plot_decision_boundary(clf, idx, X_train, X_test, y_train, y_test):
  plt.figure()
  plt.title(f'Dataset {idx}, max_depth = {clf.best_params_["max_depth"]}, n_estimators = {clf.estimator.n_estimators}')
  # plt.title(f'RFC: Dataset {idx}, max_depth = {clf.max_depth} with decision boundary')
  plt.xlabel('x1')
  plt.ylabel('x2')
  plotting.plot_decision_boundary(X_train, clf)
  plotting.plot_dataset(X_train, X_test, y_train, y_test)
  plt.show()


def __train_random_forest(X_train, X_test, y_train, y_test, idx, n_estimators):
  rf = RandomForestClassifier(n_estimators)
  param_grid = {
    "max_depth": [1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 200],
  }

  clf = GridSearchCV(rf, param_grid, n_jobs=-1)
  clf.fit(X_train, y_train)

  scores = clf.cv_results_
  best_params = clf.best_params_
  test_score = clf.score(X_test, y_test)
  mean_cv_accuracies = clf.cv_results_['mean_test_score']
  best_index = scores['params'].index(best_params)

  print(f'Dataset {idx}, estimators = {n_estimators}: The best RFC parameters found by GridSearchCV are {best_params}.')
  print(f"Dataset {idx}, estimators = {n_estimators}: Mean cross validated accuracy: {mean_cv_accuracies[best_index]:.4f}")
  print(f"Dataset {idx}, estimators = {n_estimators}: Test Score: {test_score:.4f}")

  __plot_decision_boundary(clf, idx, X_train, X_test, y_train, y_test)


if __name__ == '__main__':
  np.random.seed(RANDOM_SEED)

  for idx in [1, 2, 3]:
    X_train, X_test, y_train, y_test = get_toy_dataset(idx)

    __train_random_forest(X_train, X_test, y_train, y_test, idx, n_estimators=1)
    __train_random_forest(X_train, X_test, y_train, y_test, idx, n_estimators=100)

    # clf = RandomForestClassifier(n_estimators=100, max_depth=1000)
    # clf.fit(X_train, y_train)
















