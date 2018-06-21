import numpy as np

def gini_index(groups: np.ndarray, classes: np.ndarray) -> float:
    gini = 0

    for group in groups:
        if group.size == 0:
            continue

        score = 0
        for _class in classes:
            ratio_class = np.sum(group[:, -1] == _class) / group.size
            score += ratio_class ** 2

        gini += (1.0 - score) * (group.size / groups.size)

    return gini
