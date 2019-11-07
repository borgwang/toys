import sklearn.ensemble as ensemble
import sklearn.tree as tree
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error as mse_score
from sklearn.model_selection import train_test_split

from decision_tree import DecisionTreeClassifier
from decision_tree import DecisionTreeRegressor
from decision_tree import XGBoostDecisionTreeRegressor
from gradient_boosting import GradientBoostingClassifier
from gradient_boosting import GradientBoostingRegressor
from random_forest import RandomForestClassifier
from random_forest import RandomForestRegressor


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
    model = DecisionTreeClassifier(criterion="gini")
    model.fit(train_x, train_y)
    test_preds = model.predict(test_x)
    print("feat_importances-mine: ", model.feature_importances_)

    model = tree.DecisionTreeClassifier(criterion="gini")
    model.fit(train_x, train_y)
    test_preds2 = model.predict(test_x)
    print("feat_importances-sklearn: ", model.feature_importances_)

    print("acc-mine: %.4f" % accuracy_score(test_y, test_preds))
    print("acc-sklearn: %.4f" % accuracy_score(test_y, test_preds2))


def test_dt_regressor():
    train_x, test_x, train_y, test_y = get_regression_dataset()

    model = DecisionTreeRegressor()
    model.fit(train_x, train_y)
    test_preds = model.predict(test_x)
    print("feat_importances-mine: ", model.feature_importances_)

    model = tree.DecisionTreeRegressor()
    model.fit(train_x, train_y)
    test_preds2 = model.predict(test_x)
    print("feat_importances-sklearn: ", model.feature_importances_)

    print("mse-mine: %.4f" % mse_score(test_y, test_preds))
    print("mse-sklearn: %.4f" % mse_score(test_y, test_preds2))


def test_gbdt_classifier():
    train_x, test_x, train_y, test_y = get_classification_dataset()

    model = GradientBoostingClassifier()
    model.fit(train_x, train_y)
    test_preds = model.predict(test_x)
    print("feat_importances-mine: ", model.feature_importances_)
    print("acc-mine: %.4f" % accuracy_score(test_y, test_preds))

    train_y = train_y.ravel()
    test_y = test_y.ravel()
    model = ensemble.GradientBoostingClassifier()
    model.fit(train_x, train_y)
    test_preds2 = model.predict(test_x)
    print("feat_importances-sklearn: ", model.feature_importances_)
    print("acc-sklearn: %.4f" % accuracy_score(test_y, test_preds2))


def test_gbdt_regressor():
    train_x, test_x, train_y, test_y = get_regression_dataset()

    model = GradientBoostingRegressor(max_depth=3, n_estimators=50)
    model.fit(train_x, train_y)
    test_preds = model.predict(test_x)
    print("feat_importances-mine: ", model.feature_importances_)
    print("mse-mine: %.4f" % mse_score(test_y, test_preds))

    train_y = train_y.ravel()
    test_y = test_y.ravel()
    model = ensemble.GradientBoostingRegressor(max_depth=3, n_estimators=50)
    model.fit(train_x, train_y)
    test_preds2 = model.predict(test_x)
    print("feat_importances-sklearn: ", model.feature_importances_)
    print("mse-sklearn: %.4f" % mse_score(test_y, test_preds2))


def test_rf_classifier():
    train_x, test_x, train_y, test_y = get_classification_dataset()
    model = RandomForestClassifier(n_estimators=10)
    model.fit(train_x, train_y)
    test_preds = model.predict(test_x)
    print("feat_importances-mine: ", model.feature_importances_)

    model = ensemble.RandomForestClassifier(n_estimators=10)
    model.fit(train_x, train_y)
    test_preds2 = model.predict(test_x)
    print("feat_importances-sklearn: ", model.feature_importances_)

    print("acc-mine: %.4f" % accuracy_score(test_y, test_preds))
    print("acc-sklearn: %.4f" % accuracy_score(test_y, test_preds2))


def test_rf_regressor():
    train_x, test_x, train_y, test_y = get_regression_dataset()

    model = RandomForestRegressor(n_estimators=10)
    model.fit(train_x, train_y)
    test_preds = model.predict(test_x)
    print("feat_importances-mine: ", model.feature_importances_)

    model = ensemble.RandomForestRegressor(n_estimators=10)
    model.fit(train_x, train_y)
    test_preds2 = model.predict(test_x)
    print("feat_importances-sklearn: ", model.feature_importances_)

    print("mse-mine: %.4f" % mse_score(test_y, test_preds))
    print("mse-sklearn: %.4f" % mse_score(test_y, test_preds2))


def test_xgbdt_regressor():
    train_x, test_x, train_y, test_y = get_regression_dataset()

    model = XGBoostDecisionTreeRegressor()
    model.fit(train_x, train_y)
    test_preds = model.predict(test_x)
    print("feat_importances-mine: ", model.feature_importances_)

    model = tree.DecisionTreeRegressor()
    model.fit(train_x, train_y)
    test_preds2 = model.predict(test_x)
    print("feat_importances-sklearn: ", model.feature_importances_)

    print("mse-mine: %.4f" % mse_score(test_y, test_preds))
    print("mse-sklearn: %.4f" % mse_score(test_y, test_preds2))

if __name__ == "__main__":
    #test_dt_classifier()
    #test_dt_regressor()
    #test_gbdt_classifier()
    #test_gbdt_regressor()
    #test_rf_classifier()
    #test_rf_regressor()
    test_xgbdt_regressor()

