from collections import Counter

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


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

    def __init__(self, feat_idx=None, threshold=None, value=None, 
                 left=None, right=None):
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
        self.min_samples_split = min_samples_split
        self.min_impurity_split = min_impurity_split
        self.max_depth = max_depth
        self.loss = loss

        self.root = None

    def fit(self, X, y, loss=None):
        self.root = self._build_tree(X, y)
        self.loss = None

    def predict(self, X):
        return [self._predict_sample(x) for x in X]

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

        max_impurity = 0
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
                 max_depth=10,
                 min_samples_split=2,
                 min_impurity_split=1e-7):
        super().__init__(max_depth, min_samples_split, min_impurity_split, loss=None)
        assert criterion in ("entropy", "gini")
        self.criterion = criterion

    def _impurity_func(self, y, l_y, r_y):
        if self.criterion == "gini":
            return self._gini(y, l_y, r_y)
        else:
            return self._info_gain(y, l_y, r_y)

    def _gini(self, y):
        pass

    def _info_gain(self, y):
        def _entropy(ys):
            counts = dict(Counter(y.reshape(-1)))
            probs = np.array([1. * cnt / len(ys) for cnt in counts.values()])
            return -np.sum(probs * np.log(probs))
        before = _entropy(y)
        after = _entropy(l_y) + _entropy(r_y)
        return before - after
        
    def _aggregation_func(self, y):
        # majority vote
        res = Counter(y.reshape(-1))
        return res.most_common()[0][0]


class RegressionTree(DecisionTree):
    
    def _impurity_func(self):
        pass

    def _aggregation_func(self):
        pass


if __name__ == "__main__":
    data = load_iris()
    X = data.data
    y = data.target
    y = y.reshape((-1, 1))

    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)
    model = ClassificationTree(criterion="gini", max_depth=10, min_samples_split=2)
    model.fit(train_x, train_y)
    test_preds = model.predict(test_x)

    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(criterion="gini", max_depth=10, min_samples_split=2)
    model.fit(train_x, train_y)
    test_preds2 = model.predict(test_x)

    print("mse-mine: %.4f" % np.mean((test_preds - test_y)**2))
    print("mse-sklearn: %.4f" % np.mean((test_preds2 - test_y)**2))
