import numpy as np
import sklearn.datasets as skd
import sklearn.model_selection as skms
import sklearn.metrics as skm
import sklearn.tree as skt

from arbitrary_woodland.tree import DecisionTree


def test_decision_tree():
    X, y = skd.load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = skms.train_test_split(
        X, y, test_size=0.2
    )

    max_depth = 100
    min_size = 4
    num_features = round(float(np.sqrt(len(X[0]))))

    tree = DecisionTree(max_depth, min_size, num_features)

    tree.fit(X_train, y_train)
    pred = tree.predict(X_test)

    assert skm.accuracy_score(y_test, pred) > 0.5
