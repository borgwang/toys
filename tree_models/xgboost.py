import numpy as np

from decision_tree import DecisionTree
from gradient_boosting import GradientBoostingClassifier
from gradient_boosting import GradientBoostingRegressor
from utils import MSE


class XGBoostDecisionTreeRegressor(DecisionTree):
    
    def __init__(self,  
                 criterion="mse",
                 max_depth=None,
                 max_features=None,
                 min_samples_split=2,
                 min_impurity_split=1e-7):
        super().__init__(criterion, max_depth, max_features, 
                         min_samples_split, min_impurity_split)
        assert criterion == "mse"
        self.loss = MSE
        self._lambda = 1.0

    def fit(self, X, y, with_pred=False):
        if not with_pred:
            y_pred = np.zeros_like(y, dtype=float)
            y = np.concatenate([y, y_pred], axis=1)
        super().fit(X, y)

    def _score_func(self, y, l_y, r_y):
        y, y_pred = np.split(y, 2, axis=1)
        l_y, l_y_pred = np.split(l_y, 2, axis=1)
        r_y, r_y_pred = np.split(r_y, 2, axis=1)
        before = self.__score(y, y_pred)
        after = self.__score(l_y, l_y_pred) + self.__score(r_y, r_y_pred)
        return before - after

    def _aggregation_func(self, y):
        y, y_pred = np.split(y, 2, axis=1)
        return - self.loss.grad(y, y_pred) / (self.loss.hess(y, y_pred) + self._lambda)

    def __score(self, y, y_pred):
        G = self.loss.grad(y, y_pred)
        H = self.loss.hess(y, y_pred)
        return - 0.5 * (G ** 2 / (H + self._lambda))


class XGBoostRegressor(GradientBoostingRegressor):

    def __init__(self,
                 loss="ls",
                 learning_rate=0.1,
                 n_estimators=100,
                 criterion="mse",
                 max_features=None,
                 min_samples_split=2,
                 min_impurity_split=1e-7,
                 max_depth=None):
        super().__init__(loss, learning_rate, n_estimators, 
                         criterion, max_features, min_samples_split, 
                         min_impurity_split, max_depth)
        tree_params = {
            "criterion": criterion, 
            "max_features": max_features,
            "min_samples_split": min_samples_split,
            "min_impurity_split": min_impurity_split,
            "max_depth": max_depth}

        self.learners = [XGBoostDecisionTreeRegressor(**tree_params) 
                         for _ in range(self.n_estimators)]

    def fit(self, x, y):
        self.feature_scores_ = np.zeros(x.shape[1], dtype=float)
        self.y_dim = y.shape[1]
        F = np.zeros_like(y, dtype=float)
        for i in range(self.n_estimators):
            y_ = self._negative_grad(y, F)
            y_with_pred = np.concatenate([y_, F], axis=1)
            # fit gradient
            self.learners[i].fit(x, y_with_pred, with_pred=True)
            # update F
            f_pred = self.learners[i].predict(x)
            F += self.lr * f_pred

            # update feature importances
            self.feature_scores_ += self.learners[i].feature_scores_
        # normalize feature importances
        self.feature_importances_ = (
                self.feature_scores_ / self.feature_scores_.sum())

