from collections import Counter

import numpy as np


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
                 criterion,
                 max_depth,
                 min_samples_split, 
                 min_impurity_split,
                 loss):
        self.max_depth = float("inf") if max_depth is None else max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_split = min_impurity_split
        self.criterion = criterion
        self.loss = loss

        self.root = None

    def fit(self, X, y, loss=None):
        self.root = self._build_tree(X, y)
        self.loss = None

    def predict(self, X):
        return np.array([self._predict_sample(x) for x in X])

    def _impurity_func(self, *args, **kwargs):
        raise NotImplementedError

    def _aggregation_func(self, *args, **kwargs):
        raise NotImplementedError

    def _build_tree(self, X, y, curr_depth=0):
        n_samples, n_feats = X.shape

        impurity = 0
        if n_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            split, impurity = self._find_best_split(X, y)

        if impurity > self.min_impurity_split:
            left = self._build_tree(split["l_x"], split["l_y"], curr_depth + 1)
            right = self._build_tree(split["r_x"], split["r_y"], curr_depth + 1)
            return DTNode(feat_idx=split["feat_idx"], threshold=split["threshold"],
                          left=left, right=right)
        else:
            # leaf node
            leaf_val = self._aggregation_func(y)
            return DTNode(value=leaf_val)

    def _find_best_split(self, X, y):
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
        assert criterion in ("info_gain", "gain_ratio", "gini")
        super().__init__(criterion, max_depth, min_samples_split, 
                         min_impurity_split, loss=None)

    def _impurity_func(self, y, l_y, r_y):
        if self.criterion == "info_gain":
            return self.__info_gain(y, l_y, r_y)
        elif self.criterion == "gain_ratio":
            return self.__gain_ratio(y, l_y, r_y)
        else:
            # use 1 - gini_index to represent impurity
            # since gini_index measures purity
            return 1.0 - self.__gini_index(y, l_y, r_y)

    @staticmethod
    def __info_gain(y, l_y, r_y, with_ratio=False):

        def entropy(values):
            counts = Counter(values.reshape(-1)).values()
            probs = np.array([1. * cnt / len(values) for cnt in counts])
            return -(probs * np.log(probs)).sum()

        l_f = len(l_y) / len(y)
        r_f = len(r_y) / len(y)
        before = entropy(y)
        after = l_f * entropy(l_y) + r_f * entropy(r_y)
        info_gain = before - after

        if with_ratio:
            iv = -(l_f * np.log(l_f) + r_f * np.log(r_f))
            info_gain /= iv
        return info_gain
        
    @staticmethod
    def __gini_index(y, l_y, r_y):

        def gini(vals):
            counts = Counter(vals.reshape(-1)).values()
            probs = np.array([1. * cnt / len(vals) for cnt in counts])
            return 1.0 - np.sum(probs ** 2)

        l_f = len(l_y) / len(y)
        r_f = len(r_y) / len(y)
        return l_f * gini(l_y) + r_f * gini(r_y)

    def _aggregation_func(self, y):
        res = Counter(y.reshape(-1))
        return res.most_common()[0][0]


class RegressionTree(DecisionTree):
    
    def __init__(self, 
                 criterion="mse", 
                 max_depth=None,
                 min_samples_split=2,
                 min_impurity_split=1e-7):
        assert criterion in ("mse", "mae", "friedman_mse")
        super().__init__(criterion, max_depth, min_samples_split, 
                         min_impurity_split, loss=None)

    def _impurity_func(self, y, l_y, r_y):
        if self.criterion == "mse":
            return self.__mse(y, l_y, r_y)
        elif self.criterion == "mae":
            return self.__mae(y, l_y, r_y)
        else:
            return self.__friedman_mse(y, l_y, r_y)

    def _aggregation_func(self, y):
        return np.mean(y, axis=0)

    @staticmethod
    def __mse(y, l_y, r_y):
        l_f = len(l_y) / len(y)
        r_f = len(r_y) / len(y)
        before = np.var(y)
        after = l_f * np.var(l_y) + r_f * np.var(r_y)
        return before - after

    @staticmethod
    def __mae(y, l_y, r_y):
        l_f = len(l_y) / len(y)
        r_f = len(r_y) / len(y)
        before = np.abs(y - y.mean()).mean()
        after = (l_f * np.abs(l_y - l_y.mean()).mean() +
                 r_f * np.abs(r_y - r_y.mean()).mean())
        return before - after

    @staticmethod
    def __friedman_mse(y, l_y, r_y):
        pass
