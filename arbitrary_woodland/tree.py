import numpy as np
from arbitrary_woodland.gini import gini_impurity


class DecisionTree:
    def __init__(self, max_depth=100, min_size=4, num_features="auto"):
        self.max_depth = max_depth
        self.min_size = min_size
        self.num_features = num_features

    def fit(self, X, y):
        if self.num_features == "auto":
            self.num_features = round(float(np.sqrt(len(X[0]))))

        self.root = self._get_split(X, y)
        self._split(self.root, 1)

        return self

    def predict(self, X):
        y = np.zeros(len(X))

        node = self.root
        for i, row in enumerate(X):
            y[i] = self._predict(row, node)

        return y

    def _predict(self, row, node: dict) -> float:
        if row[node["index"]] < node["value"]:
            if isinstance(node["left"], dict):
                return self._predict(row, node["left"])
            else:
                return node["left"]
        else:
            if isinstance(node["right"], dict):
                return self._predict(row, node["right"])
            else:
                return node["right"]

    def _get_split(self, X, y) -> dict:
        classes = np.unique(y)
        node = {"index": 1000, "value": 1000, "groups": None}

        score = 1000

        indices = np.random.choice(
            len(X[0]), size=self.num_features, replace=False
        )

        for index in indices:
            for i in range(len(X)):
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
            self._split(node["left"], depth + 1)

        if len(right) <= self.min_size:
            node["right"] = self._end_node(right_y)
        else:
            node["right"] = self._get_split(right, right_y)
            self._split(node["right"], depth + 1)

    def _end_node(self, y):
        _, idx, counts = np.unique(y, return_index=True, return_counts=True)
        index = idx[np.argmax(counts)]

        return y[index]
