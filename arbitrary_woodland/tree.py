import numpy as np
from arbitrary_woodland.gini import gini_impurity


class DecisionTree:
    def __init__(self, max_depth: int, min_size: int, num_features: int):
        self.max_depth = max_depth
        self.min_size = min_size

    def fit(self, X, y) -> DecisionTree:
        self.root = self._get_split(X, y)
        self._split(root, 1)

        return self

    def _get_split(self, X, y) -> dict:
        classes = np.unique(y)
        node = {"index": 1000, "value": 1000, "groups": None}

        score = 1000

        indices = np.random.choice(len(X[0]), size=num_features, replace=False)

        for index in indices:
            for i in len(X):
                group_indices = X[:, index] < X[i, index]
                groups = (X[group_indices], X[~group_indices])
                gini = gini_impurity(groups, classes)

                if gini < score:
                    node["index"] = index
                    node["value"] = X[i, index]
                    node["groups"] = groups
                    node["groups_y"] = (y[group_indices], y[~group_indices])

        return node

    def _split(self, node: dict, depth: int) -> None:
        left, right = node["groups"]
        left_y, right_y = node["groups_y"]

        del node["groups"]
        del node["groups_y"]

        if left.size == 0 or right.size == 0:
            node["left"] = node["right"] = self._end_node(
                np.append(left_y, right_y)
            )

            return

        if depth >= self.max_depth:
            node["left"] = self._end_node(left_y)
            node["right"] = self._end_node(right_y)

            return

        if len(left) <= self.min_size:
            node["left"] = self._end_node(left_y)
        else:
            node["left"] = self._get_split(left, left_y)
            split(node["left"], depth + 1)

        if len(right) <= self.min_size:
            node["right"] = self._end_node(right_y)
        else:
            node["right"] = self._get_split(right, right_y)
            split(node["right"], depth + 1)

    def _end_node(self, y):
        _, idx, counts = np.unique(y, return_index=True, return_counts=True)
        index = idx[np.argmax(counts)]

        return y[index]
