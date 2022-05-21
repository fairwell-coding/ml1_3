import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV

import plotting
from datasets import get_toy_dataset
from task1_1 import KNearestNeighborsClassifier
from sklearn.model_selection import cross_val_score


if __name__ == '__main__':
  X_train, X_test, y_train, y_test = get_toy_dataset(2, apply_noise=True)

  for k in [1, 5, 20, 50, 100]:
    clf = KNearestNeighborsClassifier(k)
    clf.fit(X_train, y_train)

    cv_score = cross_val_score(clf, X_test, y_test, n_jobs=-1)
    print(f"Mean cross validation score for k={k}: {np.mean(cv_score)}")

    test_score = clf.score(X_test, y_test)
    print(f"Test Score for k={k}: {test_score}")

    # plot dataset with decision boundary
    plt.figure()
    plt.title(f'Noisy dataset 2 for k = {k} with decision boundary')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plotting.plot_decision_boundary(X_train, clf)
    plotting.plot_dataset(X_train, X_test, y_train, y_test)
    plt.show()

  # TODO find the best parameters for the noisy dataset!
  knn = KNearestNeighborsClassifier()
  clf = ...
  # TODO The `cv_results_` attribute of `GridSearchCV` contains useful aggregate information
  # such as the `mean_train_score` and `mean_test_score`. Plot these values as a function of `k` and report the best
  # parameters. Is the classifier very sensitive to the choice of k?
