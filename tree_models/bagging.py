from collections import Counter

import numpy as np

from decision_tree import DecisionTreeClassifier
from decision_tree import DecisionTreeRegressor


class RandomForest:
    
    def __init__(self,
                 n_estimators,
                 criterion,
                 max_features,
                 min_samples_split,
                 min_impurity_split,
                 max_depth):

        self.n_estimators = n_estimators
        self.tree_params = {
            "criterion": criterion, 
            "max_features": max_features,
            "min_samples_split": min_samples_split,
            "min_impurity_split": min_impurity_split,
            "max_depth": max_depth}
        self.learners = None

    def fit(self, X, y):
        for i in range(self.n_estimators):
            self.learners[i].fit(X, y)

    def predict(self, X):
        return np.array([l.predict(X) for l in self.learners])


class RandomForestClassifier(RandomForest):

    def __init__(self,
                 n_estimators=100,
                 criterion="gini",
                 max_features=None,
                 min_samples_split=2,
                 min_impurity_split=1e-7,
                 max_depth=None):
        super().__init__(n_estimators, criterion,  
                         max_features, min_samples_split, 
                         min_impurity_split, max_depth)
        self.learners = [DecisionTreeClassifier(**self.tree_params)
                         for _ in range(self.n_estimators)]

    def predict(self, X):
        all_preds = super().predict(X)
        res = []
        for s in all_preds.T:
            res.append(Counter(s).most_common()[0][0])
        return np.array(res)


class RandomForestRegressor(RandomForest):

    def __init__(self,
                 n_estimators=100,
                 criterion="mse",
                 max_features=None,
                 min_samples_split=2,
                 min_impurity_split=1e-7,
                 max_depth=None):
        super().__init__(n_estimators, criterion,  
                         max_features, min_samples_split, 
                         min_impurity_split, max_depth)

        self.learners = [DecisionTreeRegressor(**self.tree_params)
                         for _ in range(self.n_estimators)]

    def predict(self, X):
        return super().predict(X).mean(0)
