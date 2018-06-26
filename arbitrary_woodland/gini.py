import numpy as np
import numba


@numba.njit(cache=True)
def gini_impurity(groups: np.ndarray, classes: np.ndarray) -> float:
    gini = 0

    for i in range(len(groups)):
        group = groups[i]
        if len(group) == 0:
            continue

        score = 0
        for j in range(len(classes)):
            _class = classes[j]
            ratio_class = np.sum(group == _class) / len(group)
            score += ratio_class ** 2

        gini += 1.0 - score

    return gini
