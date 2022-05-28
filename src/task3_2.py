import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, LinearSVC

from datasets import get_heart_dataset, get_toy_dataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import RFECV
import pickle as pkl


RANDOM_SEED = 42


def __train_initial_random_forest():
  rfc = RandomForestClassifier()

  param_grid = {
    "max_depth": [5, 10, 15, 20, 25, 50, 75, 100, 200],
    "n_estimators": [1, 3, 5, 10, 25, 50, 100, 200, 250]
  }

  clf = GridSearchCV(rfc, param_grid, n_jobs=-1)
  clf.fit(X_train, y_train)

  scores = clf.cv_results_
  best_params = clf.best_params_
  test_score = clf.score(X_test, y_test)
  mean_cv_accuracies = clf.cv_results_['mean_test_score']
  best_index = scores['params'].index(best_params)

  print(f'Dataset 4: The best RFC parameters found by GridSearchCV are {best_params}.')
  print(f"Dataset 4: Mean cross validated accuracy: {mean_cv_accuracies[best_index]:.4f}")
  print(f"Dataset 4: Test Score: {test_score:.4f}")


def __train_initial_svc(X_train, X_test, y_train, y_test, subset):
  svc = SVC(tol=1e-4)

  if subset == 'full':
    param_grid = {
      "C": 10. ** np.arange(-3, 4),
      "gamma": np.logspace(-3, 3, 7),
      "kernel": ['linear', 'poly', 'rbf']
    }
  else:  # use optimal parameters from before since full grid took extremely long to calculate
    param_grid = {
      "C": [0.01],
      "gamma": [0.001],
      "kernel": ['linear']
    }

  clf = GridSearchCV(svc, param_grid, n_jobs=-1)
  clf.fit(X_train, y_train)

  scores = clf.cv_results_
  best_params = clf.best_params_
  test_score = clf.score(X_test, y_test)
  mean_cv_accuracies = clf.cv_results_['mean_test_score']
  best_index = scores['params'].index(best_params)

  print(f'Dataset 4 - {subset}: The best SVC parameters found by GridSearchCV are {best_params}.')
  print(f"Dataset 4 - {subset}: Mean cross validated accuracy: {mean_cv_accuracies[best_index]:.4f}")
  print(f"Dataset 4 - {subset}: Test Score: {test_score:.4f}")


def __find_most_important_features():
  rfc = RandomForestClassifier(max_depth=20, n_estimators=50)  # use best parameters found by GridSearchCV
  rfc.fit(X_train, y_train)

  feature_indices_desc = np.flip(np.argsort(rfc.feature_importances_))

  print('Most important features found by random forest classifier: ')
  for i in range(0, 7):
    print(f'{feature_indices_desc[i]}: {rfc.feature_importances_[feature_indices_desc[i]]:.4f}')

  __plot_relative_feature_importance(rfc)

  return rfc


def __plot_relative_feature_importance(rfc):
  plt.barh(range(0, 25), rfc.feature_importances_)
  plt.title("Relative feature importance")
  plt.xlabel("feature importance")
  plt.ylabel("feature index")
  plt.show()


if __name__ == '__main__':
  np.random.seed(RANDOM_SEED)

  X_train, X_test, y_train, y_test = get_toy_dataset(4)

  __train_initial_random_forest()
  __train_initial_svc(X_train, X_test, y_train, y_test, 'full')
  rfc = __find_most_important_features()

  rfecv = RFECV(rfc, scoring='accuracy')
  rfecv.fit(X_train, y_train)

  __train_initial_svc(rfecv.transform(X_train), rfecv.transform(X_test), y_train, y_test, 'optimal')
