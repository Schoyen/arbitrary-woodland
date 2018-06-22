import numpy as np
import sklearn.datasets as skd
import sklearn.model_selection as skms
import sklearn.metrics as skm
import sklearn.tree as skt

from arbitrary_woodland.tree import DecisionTree


def test_decision_tree():
    X, y = skd.load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = skms.train_test_split(
        X, y, test_size=0.2
    )

    max_depth = 100
    min_size = 1
    num_features = 4

    tree = DecisionTree(max_depth, min_size, num_features)

    tree.fit(X_train, y_train)
    pred = tree.predict(X_test)
    print(skm.accuracy_score(y_test, pred))

    tree = skt.DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_size,
        max_features=num_features,
    )
    tree.fit(X_train, y_train)
    pred = tree.predict(X_test)
    print(skm.accuracy_score(y_test, pred))
    wat
