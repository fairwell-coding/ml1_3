import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV

import plotting
from datasets import get_toy_dataset
from task1_1 import KNearestNeighborsClassifier

if __name__ == '__main__':
  for idx in [1, 2, 3]:
    X_train, X_test, y_train, y_test = get_toy_dataset(idx)
    knn = KNearestNeighborsClassifier()

    param_grid = {
        "k": range(1, 101)
    }

    clf = GridSearchCV(knn, param_grid, n_jobs=-1, return_train_score=True)
    clf.fit(X_train, y_train)

    test_score = clf.score(X_test, y_test)

    print(f"Test Score: {test_score}")
    print(f"Dataset {idx}: {clf.best_params_}")

    # plot dataset with decision boundary
    plt.figure()
    plt.title(f'Dataset {idx} with decision boundary')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plotting.plot_decision_boundary(X_train, clf)
    plotting.plot_dataset(X_train, X_test, y_train, y_test)
    plt.show()

    # plot training and validation score for all values of k
    plt.figure()
    plt.title(f'Dataset {idx}: training and validation scores')
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.plot(clf.cv_results_['mean_train_score'], label='mean training scores')
    plt.plot(clf.cv_results_['mean_test_score'], label='mean validation scores')
    plt.legend()
    plt.show()
