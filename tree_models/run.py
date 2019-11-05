import time

import sklearn.tree as tree
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error as mse_score
from sklearn.model_selection import train_test_split

from decision_tree import DecisionTreeClassifier
from decision_tree import DecisionTreeRegressor
from gradient_boosting import GradientBoostingClassifier
from gradient_boosting import GradientBoostingRegressor


def get_classification_dataset():
    data = load_iris()
    X, y = data.data, data.target
    y = y.reshape((-1, 1))
    return train_test_split(X, y, test_size=0.3)


def get_regression_dataset():
    data = load_boston()
    X, y = data.data, data.target
    y = y.reshape((-1, 1))
    return train_test_split(X, y, test_size=0.3)


def test_dt_classifier():
    train_x, test_x, train_y, test_y = get_classification_dataset()
    ts = time.time()
    model = GradientBoostingClassifier()
    model.fit(train_x, train_y)
    test_preds = model.predict(test_x)
    te = time.time()

    ts2 = time.time()
    model = tree.GradientBoostingClassifier()
    model.fit(train_x, train_y)
    te2 = time.time()
    test_preds2 = model.predict(test_x)

    print("acc-mine: %.4f" % accuracy_score(test_y, test_preds))
    print("acc-sklearn: %.4f" % accuracy_score(test_y, test_preds2))
    print((te - ts) / (te2 - ts2))


def test_dt_regressor():
    train_x, test_x, train_y, test_y = get_classification_dataset()

    ts = time.time()
    model = GradientBoostingRegressor()
    model.fit(train_x, train_y)
    te = time.time()
    test_preds = model.predict(test_x)

    ts2 = time.time()
    model = tree.GradientBoostingRegressor()
    model.fit(train_x, train_y)
    te2 = time.time()
    test_preds2 = model.predict(test_x)

    print("mse-mine: %.4f" % mse_score(test_y, test_preds))
    print("mse-sklearn: %.4f" % mse_score(test_y, test_preds2))
    print((te - ts) / (te2 - ts2))


def test_gbdt_classifier():
    train_x, test_x, train_y, test_y = get_classification_dataset()
    ts = time.time()
    model = DecisionTreeRegressor(criterion="mse")
    model.fit(train_x, train_y)
    te = time.time()
    test_preds = model.predict(test_x)


if __name__ == "__main__":
    #test_dt_classifer()
    #test_dt_regressor()
    test_gbdt_classifier()
    #test_gbdt_regressor()
