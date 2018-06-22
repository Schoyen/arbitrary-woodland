import numpy as np
import sklearn.datasets as skd
import sklearn.model_selection as skms
import sklearn.metrics as skm
import sklearn.ensemble as ske

from arbitrary_woodland.woodland import ArbitraryWoodland


def test_forest():
    X, y = skd.load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = skms.train_test_split(
        X, y, test_size=0.2
    )

    num_trees = 10
    sample_ratio = 0.7

    max_depth = 100
    min_size = 4
    num_features = round(float(np.sqrt(len(X[0]))))

    args = [max_depth, min_size, num_features]

    forest = ArbitraryWoodland(num_trees, sample_ratio, *args)

    forest.fit(X_train, y_train)
    pred = forest.predict(X_test)

    assert skm.accuracy_score(y_test, pred) > 0.5
