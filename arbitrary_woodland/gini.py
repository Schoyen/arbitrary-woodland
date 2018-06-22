import numpy as np


def gini_impurity(groups: np.ndarray, classes: np.ndarray) -> float:
    gini = 0

    for group in groups:
        if len(group) == 0:
            continue

        score = 0
        for _class in classes:
            ratio_class = np.sum(group == _class) / len(group)
            score += ratio_class ** 2

        gini += 1.0 - score

    return gini
