from collections import Counter

import numpy as np

from utils import is_numerical


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
                 max_features,
                 min_samples_split, 
                 min_impurity_split):
        self.max_depth = float("inf") if max_depth is None else max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_impurity_split = min_impurity_split
        self.criterion = criterion

        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

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

        # subset of feature columns
        k = self._get_n_feats(self.max_features, n_feats)
        cols = np.random.choice(range(n_feats), k, replace=False)

        for col in cols:
            # for each feature
            for thr in np.unique(X[:, col]):
                # for each unique value of curr feature
                Xy1, Xy2 = self._divide(Xy, col, thr)
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

    @staticmethod
    def _divide(data, col, thr):
        if is_numerical(thr):
            mask = data[:, col] < thr
        else:
            mask = data[: col] == thr
        return data[mask], data[~mask]

    @staticmethod
    def _get_n_feats(max_feats, n_feats):
        if isinstance(max_feats, int):
            return max_feats
        elif isinstance(max_feats, float):
            return int(max_feats * n_feats)
        elif isinstance(max_feats, str):
            if max_feats == "sqrt":
                return int(np.sqrt(n_feats))
            elif max_feats == "log2":
                return int(np.log2(n_feats))
        return n_feats


class DecisionTreeClassifier(DecisionTree):

    def __init__(self, 
                 criterion="gini", 
                 max_depth=None,
                 max_features=None,
                 min_samples_split=2,
                 min_impurity_split=1e-7):
        assert criterion in ("info_gain", "gain_ratio", "gini")
        super().__init__(criterion, max_depth, max_features, 
                         min_samples_split, min_impurity_split)

    def _impurity_func(self, y, l_y, r_y):
        if self.criterion == "info_gain":
            return self.__info_gain(y, l_y, r_y)
        elif self.criterion == "gain_ratio":
            return self.__info_gain(y, l_y, r_y, with_ratio=True)
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


class DecisionTreeRegressor(DecisionTree):

    def __init__(self, 
                 criterion="friedman_mse", 
                 max_depth=None,
                 max_features=None,
                 min_samples_split=2,
                 min_impurity_split=1e-7):
        assert criterion in ("mse", "mae", "friedman_mse")
        super().__init__(criterion, max_depth, max_features, 
                         min_samples_split, min_impurity_split)

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
        mse_loss = lambda x: np.mean(np.square(x - x.mean(0)), 0)
        l_f = len(l_y) / len(y)
        before = mse_loss(y)
        after = l_f * mse_loss(l_y) + (1 - l_f) * mse_loss(r_y)
        return np.mean(before - after)

    @staticmethod
    def __mae(y, l_y, r_y):
        mae_loss = lambda x: np.mean(np.abs(x - x.mean(0)), 0)
        l_f = len(l_y) / len(y)
        before = mae_loss(y)
        after = l_f * mae_loss(l_y) + (1 - l_f) * mae_loss(r_y)
        return np.mean(before - after)

    @staticmethod
    def __friedman_mse(y, l_y, r_y):
        l_mean, r_mean = l_y.mean(0), r_y.mean(0)
        friedman_mse = len(l_y) * len(r_y) * (l_mean - r_mean) ** 2 / len(y)
        return np.mean(friedman_mse)
