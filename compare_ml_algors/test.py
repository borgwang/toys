from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import svm
from sklearn import ensemble
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, log_loss
from sklearn.grid_search import GridSearchCV

from k_flod_val import KFlod


numerical_field_idx = [0, 2, 4, 10, 11, 12]
categorical_field_idx = [1, 3, 5, 6, 7, 8, 9, 13]
label_idx = 14
num_features = 14


def data_preprocess(data_path):
    data = []
    for line in open(data_path):
        data.append(line.strip('\n').strip('.').split(', '))

    data = np.array(data)

    numerical_fields, categorical_fields = [], []
    for i in numerical_field_idx:
        numerical_fields.append(data[:, i])
    numerical_features = np.asarray(numerical_fields, dtype=np.float).T
    for idx in categorical_field_idx:
        mapping = dict()
        for i, c in enumerate(sorted(list(set(data[:, idx])))):
            if c != '?':
                mapping[c] = i
        # pad with most-frequent element
        mapping['?'] = None
        field = np.asarray([mapping[d] for d in data[:, idx]])
        field[field == None] = Counter(field).most_common(1)[0][0]
        categorical_fields.append(field)

    categorical_features = np.asarray(categorical_fields, dtype=np.float).T
    features = np.hstack((numerical_features, categorical_features))

    # normalize
    features = preprocessing.minmax_scale(features)
    features -= np.mean(features)
    features /= np.std(features)
    # features = preprocessing.normalize(features)

    label_mapping = {'>50K': 1.0, '<=50K': -1.0}
    labels = [label_mapping[i] for i in data[:, label_idx]]
    labels = np.array(labels, dtype=np.float).reshape(-1, 1)

    return features, labels


k = 10
X, Y = data_preprocess('./adult_train.data')
# k_flod = KFlod(X, Y, k)

# ----- Linear SVM -----
# clf = svm.LinearSVC(verbose=True, max_iter=10000, tol=1e-4, C=1.0)
# clf.fit(data['train']['inputs'], data['train']['labels'])
# predictions = clf.predict(data['val'])
# accuracy = np.sum(
#     predictions == np.ravel(data['val']['labels'])) / data['val']['size']
# print('##### accu:%.4f' % accuracy)

# ----- Kernel SVM -----
# # best hyperparameters C: 3.0, gamma: 0.04 (grid search)
# clf = svm.SVC(
#     kernel='rbf', gamma=0.04, verbose=True, max_iter=50000, tol=1e-5, C=3.0)
# clf.fit(data['train']['inputs'], data['train']['labels'])
# predictions = clf.predict(data['val']['inputs'])
# accuracy = np.sum(
#     predictions == np.ravel(data['val']['labels'])) / data['val']['size']
# print('##### accu:%.4f' % accuracy)

# ----- Random Forest -----
test_x, test_y = data_preprocess('./adult_test.data')
clf = ensemble.RandomForestClassifier()
y = np.ravel(Y)
# clf.fit(X, y)
# y_prob = clf.predict_proba(test_x)
# print(log_loss(test_y, y_prob))
# grid search
param_test1 = [{'n_estimators': range(10, 101, 10)}]
gsearch1 = GridSearchCV(
    estimator=clf, param_grid=param_test1, scoring='accuracy', cv=5)
gsearch1.fit(X, y)
print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)
