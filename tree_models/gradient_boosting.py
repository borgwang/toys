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

        self.learners = [DecisionTreeRegressor(**tree_params) 
                         for _ in range(self.n_estimators)]

    def fit(self, X, y):
        self.y_dim = y.shape[1]
        F = np.zeros_like(y, dtype=float)
        for i in range(self.n_estimators):
            grads = self._gradient_func(y, F)
            # fit gradient
            self.learners[i].fit(X, grads)
            # update F
            grads_preds = self.learners[i].predict(X)
            F -= self.lr * grads_preds

    def predict(self, X):
        F = np.zeros((X.shape[0], self.y_dim), dtype=float)
        for i, learner in enumerate(self.learners):
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
        return softmax(F) - y

    @staticmethod
    def __exponential_grad(y, F):
        # TODO
        pass

    def fit(self, X, y):
        y = get_one_hot(y, len(np.unique(y)))
        super().fit(X, y)

    def predict(self, X):
        preds = super().predict(X)
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

        grad_func_dict = {"ls": self.__ls_grad,
                          "lad": self.__lad_grad,
                          "huber": self.__huber_grad,
                          "quantile": self.__quantile_grad}
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
