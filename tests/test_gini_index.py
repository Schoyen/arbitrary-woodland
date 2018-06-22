import numpy as np

from arbitrary_woodland.gini import gini_impurity


def test_gini_impurity():
    classes = np.array([1, 2, 3])
    groups = np.array([np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3])])

    gini = gini_impurity(groups, classes)

    assert abs(gini - 0.66) < 1e-8


def test_gini_impurity_2():
    classes = np.array([1, 2])
    groups = np.array([np.array([1, 2, 2, 2])])

    gini = gini_impurity(groups, classes)

    assert abs(gini - 3 / 8) < 1e-8
