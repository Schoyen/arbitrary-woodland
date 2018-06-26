import numpy as np
from arbitrary_woodland.tree import DecisionTree


class ArbitraryWoodland:
    def __init__(self, num_trees=10, sample_ratio=0.7, **kwargs) -> None:
        self.kwargs = kwargs
        self.num_trees = num_trees
        self.sample_ratio = sample_ratio

    def fit(self, X, y):
        self.trees = []

        for i in range(self.num_trees):
            _X, _y = self._subsample(X, y, self.sample_ratio)
            tree = DecisionTree(**self.kwargs).fit(_X, _y)
            self.trees.append(tree)

        return self

    def predict(self, X):
        y = np.zeros(len(X))
        predictions = np.array([tree.predict(X) for tree in self.trees])

        for i, row in enumerate(predictions.T):
            _, idx, counts = np.unique(
                row, return_index=True, return_counts=True
            )

            index = idx[np.argmax(counts)]
            y[i] = row[index]

        return y

    def _subsample(self, X, y, ratio: float) -> tuple:
        num_samples = round(len(X) * ratio)
        indices = np.random.randint(len(X), size=num_samples)

        return X[indices], y[indices]
