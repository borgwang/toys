import numpy as np

from decision_tree import DecisionTreeClassifier
from decision_tree import DecisionTreeRegressor


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

        self.learners = [DecisionTreeClassifier(**tree_params) 
                         for _ in range(self.n_estimators)]

    def fit(self, X, y):
        self.init_F = np.ones_like(y) * y.mean(0)
        F = self.init_F
        for i in range(self.n_estimators):
            # fit gradient
            grads = self._gradient_func(y, F)
            self.learners[i].fit(X, grads)
            grads_preds = self.learners[i].predict(X)
            # update F
            F -= self.lr * grads_preds

    def predict(self, X):
        F = self.init_F
        for learner in self.learners:
            grads_preds = learner.predict(X)
            F -= self.lr * grads_preds
        return F

    def _gradient_func(self, y, F):
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

        grad_func_dict = {"deviance": self.__deviance_grad,
                          "exponential": self.__exponential_grad}
        assert loss in grad_func_dict
        self._gradient_func = grad_func_dict[loss]

    @staticmethod
    def __deviance_grad(y, F):
        logistic = lambda x: 1.0 / (1.0 + np.exp(-x))
        return logistic(F) - y

    @staticmethod
    def __exponential_grad():
        pass


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

        grad_func_dict = {"ls": self.__ls_grad,
                          "lad": self.__lad_grad,
                          "huber": self.__huber_grad,
                          "lad": self.__lad_grad}
        assert loss in grad_func_dict
        self._gradient_func = grad_func_dict[loss]

    @staticmethod
    def __ls_grad(y, F):
        return F - y

    @staticmethod
    def __lad_grad(y, F):
        pass

    @staticmethod
    def __huber_grad(y, F):
        pass

    @staticmethod
    def __quantile_grad(y, F):
        pass
