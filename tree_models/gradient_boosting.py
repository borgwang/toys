import numpy as np

from decision_tree import DecisionTreeRegressor
from decision_tree import XGBoostDecisionTreeRegressor
from utils import get_one_hot
from utils import softmax


class GradientBoosting:
    
    def __init__(self,
                 loss,
                 learning_rate,
                 n_estimators,
                 criterion,
                 max_features,
                 min_samples_split,
                 min_impurity_split,
                 max_depth):
        self.loss = loss
        self.lr = learning_rate
        self.n_estimators = n_estimators
        tree_params = {
            "criterion": criterion, 
            "max_features": max_features,
            "min_samples_split": min_samples_split,
            "min_impurity_split": min_impurity_split,
            "max_depth": max_depth}

        self.learners = [DecisionTreeRegressor(**tree_params) 
                         for _ in range(self.n_estimators)]
        self.feature_importances_ = None
        self.feature_scores_ = None

    def fit(self, x, y):
        self.feature_scores_ = np.zeros(x.shape[1], dtype=float)
        self.y_dim = y.shape[1]
        F = np.zeros_like(y, dtype=float)
        for i in range(self.n_estimators):
            # fit gradient
            self.learners[i].fit(x, y=self._negative_grad(y, F))
            # update F
            f_pred = self.learners[i].predict(x)
            F += self.lr * f_pred
            # update feature importances
            self.feature_scores_ += self.learners[i].feature_scores_

        # normalize feature importances
        self.feature_importances_ = (
                self.feature_scores_ / self.feature_scores_.sum())

    def predict(self, x):
        F = np.zeros((x.shape[0], self.y_dim), dtype=float)
        for i, learner in enumerate(self.learners):
            f_pred = learner.predict(x)
            F += self.lr * f_pred
        return F

    def _negative_grad(self, y, F):
        raise NotImplementedError


class GradientBoostingClassifier(GradientBoosting):

    def __init__(self,
                 loss="deviance",
                 learning_rate=0.1,
                 n_estimators=100,
                 criterion="friedman_mse",
                 max_features=None,
                 min_samples_split=2,
                 min_impurity_split=1e-7,
                 max_depth=None):
        super().__init__(loss, learning_rate, n_estimators, 
                         criterion, max_features, min_samples_split, 
                         min_impurity_split, max_depth)

        func_dict = {"deviance": self.__deviance,
                     "exponential": self.__exponential}
        assert loss in func_dict
        self._negative_grad = func_dict[loss]

    @staticmethod
    def __deviance(y, F):
        return y - softmax(F)

    @staticmethod
    def __exponential(y, F):
        # TODO
        pass

    def fit(self, x, y):
        y = get_one_hot(y, len(np.unique(y)))
        super().fit(x, y)

    def predict(self, x):
        preds = super().predict(x)
        probs = softmax(preds)
        return np.argmax(probs, 1).reshape(-1, 1)


class GradientBoostingRegressor(GradientBoosting):
    
    def __init__(self,
                 loss="ls",
                 learning_rate=0.1,
                 n_estimators=100,
                 criterion="friedman_mse",
                 max_features=None,
                 min_samples_split=2,
                 min_impurity_split=1e-7,
                 max_depth=None):
        super().__init__(loss, learning_rate, n_estimators, 
                         criterion, max_features, min_samples_split, 
                         min_impurity_split, max_depth)

        func_dict = {"ls": self.__ls,
                     "lad": self.__lad,
                     "huber": self.__huber,
                     "quantile": self.__quantile}
        assert loss in func_dict
        self._negative_grad = func_dict[loss]

    @staticmethod
    def __ls(y, F):
        return y - F

    @staticmethod
    def __lad(y, F):
        pass

    @staticmethod
    def __huber(y, F):
        pass

    @staticmethod
    def __quantile(y, F):
        pass
