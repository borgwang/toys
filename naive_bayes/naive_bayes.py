import numpy as np 


class NaiveBayes:

    def __init__(self):
        self.params = {}

    def fit(self, x, y):
        classes = np.unique(y)
        for cls in classes:
            params = []
            x_c = x[y == cls]
            self.params[cls] = self._paramerize(cls, x_c)
        # calcualte priors P(Y)
        self.priors = self._calculate_priors(y, classes)
        self.classes = classes

    def predict(self, x):
        posteriors = []
        for cls in self.classes:
            posterior = self.priors[cls]
            for i in range(x.shape[1]):
                params = self.params[cls][i]
                likelihood = self._calculate_likelihood(x[:, i], params)
                posterior *= likelihood
            posteriors.append(posterior)
        posteriors = np.asarray(posteriors).T
        return np.argmax(posteriors, axis=1)

    @staticmethod
    def _calculate_priors(y, classes):
        return {cls: np.mean(y == cls) for cls in classes}

    @staticmethod
    def _calculate_likelihood(x, params):
        raise NotImplementedError

    def _paramerize(self, *args, **kwargs):
        raise NotImplementedError


class GaussianNaiveBayes(NaiveBayes):

    def _paramerize(self, cls, x_c):
        params = []
        for feat in x_c.T:
            params.append({"mean": feat.mean(), "var": feat.var()})
        return params

    @staticmethod
    def _calculate_likelihood(x, params):
        # gaussian likelihood estimation
        mean, var = params["mean"], params["var"]
        return (np.exp(-(x - mean) ** 2 / (2 * var)) / 
                np.sqrt(2 * np.pi * var))


class MultinomialNaiveBayes(NaiveBayes):

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = 1.0

    def _paramerize(self, cls, x_c):
        params = []
        for feat in x_c.T:
            freq = {}
            unique_vals = np.unique(feat)
            for v in unique_vals:
                p = ((feat == v).sum() + self.alpha) / (len(feat) + len(unique_vals))
                freq[v] = p
            freq[-1] = 1 / (len(feat) + len(unique_vals))
            params.append(freq)
        return params

    @staticmethod
    def _calculate_likelihood(x, params):
        # multinomial likelihood estimation
        return np.array([params.get(v, params[-1]) for v in x])
