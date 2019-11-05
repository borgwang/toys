from collections import Counter

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def is_numerical(val):
    return isinstance(val, int) or isinstance(val, float)


def divide_on_feature(Xy, feat_idx, thr):
    if is_numerical(thr):
        # numerical feature
        mask = Xy[:, feat_idx] < thr
    else:
        # categorical feature
        mask = Xy[: feat_idx] == thr
    return Xy[mask], Xy[~mask]



class DTNode:

    def __init__(self, 
                 feat_idx=None, 
                 threshold=None, 
                 left=None, 
                 right=None,
                 value=None):
        self.feat_idx = feat_idx
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right


class DecisionTree:

    def __init__(self, 
                 max_depth,
                 min_samples_split, 
                 min_impurity_split,
                 loss):
        self.max_depth = float("inf") if max_depth is None else max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_split = min_impurity_split
        self.loss = loss


        self.root = None

    def fit(self, X, y, loss=None):
        self.root = self._build_tree(X, y)
        self.loss = None

    def predict(self, X):
        return np.array([self._predict_sample(x) for x in X]).reshape(-1, 1)

    def impurity_func(self, *args, **kwargs):
        raise NotImplementedError

    def aggregation_func(self, *args, **kwargs):
        raise NotImplementedError

    def _build_tree(self, X, y, curr_depth=0):
        n_samples, n_feats = X.shape

        impurity = 0
        if n_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            split, impurity = self._find_split_point(X, y)

        if impurity > self.min_impurity_split:
            left = self._build_tree(split["l_x"], split["l_y"], curr_depth + 1)
            right = self._build_tree(split["r_x"], split["r_y"], curr_depth + 1)
            return DTNode(feat_idx=split["feat_idx"], threshold=split["threshold"],
                          left=left, right=right)
        else:
            # leaf node
            leaf_val = self._aggregation_func(y)
            return DTNode(value=leaf_val)

    def _find_split_point(self, X, y):
        # find the best feature and the best split point (largest impurity)
        Xy = np.concatenate((X, y), axis=1)
        n_feats = X.shape[1]

        max_impurity = 0.0
        best_split = None
        for col in range(n_feats):
            # for each feature
            for thr in np.unique(X[:, col]):
                # for each unique value of curr feature
                Xy1, Xy2 = divide_on_feature(Xy, col, thr)
                if not len(Xy1) or not len(Xy2):
                    continue

                l_y, r_y = Xy1[:, n_feats:], Xy2[:, n_feats:]
                impurity = self._impurity_func(y, l_y, r_y)
                if impurity > max_impurity:
                    max_impurity = impurity
                    best_split = {
                        "feat_idx": col, "threshold": thr,
                        "l_x": Xy1[:, :n_feats], "r_x": Xy2[:, :n_feats],
                        "l_y": l_y, "r_y": r_y}
        return best_split, max_impurity

    def _predict_sample(self, x, node=None):
        if node is None:
            node = self.root

        # return value for leaf nodes
        if node.value is not None:
            return node.value

        feat = x[node.feat_idx]
        if is_numerical(feat):
            node = node.left if feat < node.threshold else node.right
        else:
            node = node.left if feat == node.threshold else node.right 
        return self._predict_sample(x, node=node)


class ClassificationTree(DecisionTree):

    def __init__(self, 
                 criterion="gini", 
                 max_depth=None,
                 min_samples_split=2,
                 min_impurity_split=1e-7):
        super().__init__(max_depth, min_samples_split, min_impurity_split, loss=None)
        assert criterion in ("info_gain", "gain_ratio", "gini")
        self.criterion = criterion

    def _impurity_func(self, y, l_y, r_y):
        if self.criterion == "info_gain":
            return self.__info_gain(y, l_y, r_y)
        elif self.criterion == "gain_ratio":
            return self.__gain_ratio(y, l_y, r_y)
        else:
            # use 1-gini_index to represent impurity since gini_index measures purity
            return 1.0 - self.__gini_index(y, l_y, r_y)

    def __info_gain(self, y, l_y, r_y):

        def entropy(vals):
            cnts = dict(Counter(vals.reshape(-1)))
            probs = np.array([1. * cnt / len(vals) for cnt in cnts.values()])
            return -(probs * np.log(probs)).sum()

        before = entropy(y)
        after = (len(l_y) * entropy(l_y) + len(r_y) * entropy(r_y)) / len(y)
        return before - after
        
    def __gain_ratio(self, y, l_y, r_y):
        info_gain = self.__info_gain(y, l_y, r_y)
        # compute IV
        l_f = len(l_y) / len(y)
        r_f = len(r_y) / len(y)
        iv = -(l_f * np.log(l_f) + r_f * np.log(r_f))
        return info_gain / iv

    def __gini_index(self, y, l_y, r_y):

        def gini(vals):
            cnts = Counter(vals.reshape(-1)).values()
            probs = np.array([1. * cnt / len(vals) for cnt in cnts])
            return 1.0 - np.sum(probs ** 2)

        l_f = len(l_y) / len(y)
        r_f = len(r_y) / len(y)
        return l_f * gini(l_y) + r_f * gini(r_y)

    def _aggregation_func(self, y):
        # majority vote
        res = Counter(y.reshape(-1))
        return res.most_common()[0][0]


class RegressionTree(DecisionTree):
    
    def _impurity_func(self):
        pass

    def _aggregation_func(self):
        # simple average
        pass


if __name__ == "__main__":
    data = load_iris()
    X = data.data
    y = data.target
    y = y.reshape((-1, 1))

    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)
    model = ClassificationTree(criterion="gini", max_depth=None, min_samples_split=2)
    model.fit(train_x, train_y)
    test_preds = model.predict(test_x)

    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(criterion="gini", max_depth=None, min_samples_split=2)
    model.fit(train_x, train_y)
    test_preds2 = model.predict(test_x)

    print("acc-mine: %.4f" % accuracy_score(test_y, test_preds))
    print("acc-sklearn: %.4f" % accuracy_score(test_y, test_preds2))
