import time

from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error as mse_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from decision_tree import ClassificationTree
from decision_tree import RegressionTree


def test_classification():
    data = load_iris()
    X, y = data.data, data.target
    y = y.reshape((-1, 1))
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)

    ts = time.time()
    model = ClassificationTree(criterion="gini")
    model.fit(train_x, train_y)
    test_preds = model.predict(test_x)
    te = time.time()

    ts2 = time.time()
    model = DecisionTreeClassifier(criterion="gini")
    model.fit(train_x, train_y)
    te2 = time.time()
    test_preds2 = model.predict(test_x)

    print("acc-mine: %.4f" % accuracy_score(test_y, test_preds))
    print("acc-sklearn: %.4f" % accuracy_score(test_y, test_preds2))
    print((te - ts) / (te2 - ts2))


def test_regression():
    import numpy as np
    data = load_boston()
    X, y = data.data, data.target
    y = y.reshape((-1, 1))
    y = np.concatenate([y, y * 0.1], axis=1)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)

    ts = time.time()
    model = RegressionTree(criterion="mae")
    model.fit(train_x, train_y)
    te = time.time()
    test_preds = model.predict(test_x)

    ts2 = time.time()
    model = DecisionTreeRegressor(criterion="mae")
    model.fit(train_x, train_y)
    te2 = time.time()
    test_preds2 = model.predict(test_x)

    print("mse-mine: %.4f" % mse_score(test_y, test_preds))
    print("mse-sklearn: %.4f" % mse_score(test_y, test_preds2))
    print((te - ts) / (te2 - ts2))


if __name__ == "__main__":
    test_classification()
    test_regression()
