import numpy as np
cimport numpy as np

from arbitrary_woodland.gini import gini_impurity

np.import_array()

def get_split(
        np.ndarray[double, ndim=2] X,
        np.ndarray[long, ndim=1] y,
        int num_features
):
    cdef np.ndarray[long, ndim=1] classes, indices
    cdef dict node
    cdef double score, gini
    cdef int index, i
    cdef tuple groups

    classes = np.unique(y)
    node = {"index": 1000, "value": 1000, "groups": None}

    score = 1000

    indices = np.random.choice(
        len(X[0]), size=num_features, replace=False
    )

    for index in indices:
        for i in range(len(X)):
            group_indices = X[:, index] < X[i, index]
            groups = (y[group_indices], y[~group_indices])
            gini = gini_impurity(groups, classes)

            if gini < score:
                node["index"] = index
                node["value"] = X[i, index]
                node["groups"] = (X[group_indices], X[~group_indices])
                node["groups_y"] = (y[group_indices], y[~group_indices])

    return node
