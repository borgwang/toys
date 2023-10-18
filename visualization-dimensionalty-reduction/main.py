import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE


def prepare_data():
  iris = load_iris()
  X = iris["data"]
  X -= np.mean(X)
  X /= np.std(X)
  Y = iris["target"]
  num_features = X.shape[1]
  num_labels = len(set(Y))
  return X, Y, num_features, num_labels

def pca(X, alg="svd"):
  # eig_decomposition appraoch
  def eig_decomposition(X):
    # covarience matrix
    cov = np.cov(X.T)
    # calculate eig values and eig vectors
    eig_vals, eig_vecs = np.linalg.eig(cov)
    topk_idx = np.argsort(eig_vals)[::-1][:2]
    # construct projection matrix
    projection_matrix = eig_vecs[:, topk_idx]
    reduced_X = np.dot(X, projection_matrix)
    return reduced_X

  # svd appraoch
  def svd(X):
    U, sigma, _ = np.linalg.svd(X)
    topk = 2
    reduced_X = U[:, :topk].dot(np.diag(sigma[:topk]))
    return reduced_X

  return eig_decomposition(X) if alg == "eig" else svd(X)

def autoencoder(X):

  def build_net():
    data_input = tf.placeholder(shape=[None, 4], dtype=tf.float32)
    reduced_features = slim.fully_connected(inputs=data_input,
                                            num_outputs=2,
                                            activation_fn=tf.nn.sigmoid,
                                            scope="encoder")
    output = slim.fully_connected(inputs=reduced_features,
                                  num_outputs=4,
                                  activation_fn=None,
                                  scope="decoder")
    loss = tf.losses.mean_squared_error(data_input, output)
    train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
    return data_input, reduced_features, loss, train_op

  with tf.Session() as sess:
    data_input, reduced_features, loss, train_op = build_net()
    sess.run(tf.global_variables_initializer())
    for _ in range(3000):
      sess.run([train_op, loss], {data_input: X})
    # generate reduced data
    reduced_X = sess.run(reduced_features, {data_input: X})
    return reduced_X


def tsne(X):
  model = TSNE(learning_rate=100, n_components=2, random_state=0, perplexity=30)
  return model.fit_transform(X)

def lda(X, Y):
  # ref:https://zhuanlan.zhihu.com/p/27914876?group_id=869972703057162240
  # calculate mean X
  num_cls = len(set(Y))
  num_features = X.shape[1]
  mean_cls = []
  for c in range(num_cls):
    mean_cls.append(np.mean(X[Y == c], 0))
  # calculate within-class scatter matrix
  s_w = np.zeros((num_features, num_features))
  for c, mv in enumerate(mean_cls):
    cls_sctter_matrix = np.zeros((num_features, num_features))
    for row in X[Y == c]:
      row, mv = row.reshape(4, 1), mv.reshape(4, 1)
      cls_sctter_matrix += np.dot(row - mv, (row - mv).T)
    s_w += cls_sctter_matrix

  # calculate between-class scatter matrix
  overall_mean = np.mean(X, 0).reshape(4, 1)
  s_b = np.zeros((num_features, num_features))
  for c, mv in enumerate(mean_cls):
    num = len(X[Y == c])
    mv = mv.reshape(4, 1)
    s_b += num * np.dot(mv - overall_mean, (mv - overall_mean).T)

  # eig decomposition
  eig_vals, eig_vecs = np.linalg.eig(np.dot(np.linalg.inv(s_w), s_b))
  topk = 2
  topk_idx = np.argsort(eig_vals)[::-1][:topk]
  projection_matrix = eig_vecs[:, topk_idx]
  reduced_X = np.dot(X, projection_matrix)
  return reduced_X


if __name__ == "__main__":
  # prepare data
  X, Y, num_features, num_labels = prepare_data()

  # different algorithms
  reduced_data = {
    "PCA(svd)": pca(X, alg="svd"),
    "Autoencoder": autoencoder(X),
    "t-SNE": tsne(X),
    "LDA": lda(X, Y)
  }

  # plot results
  color_mapping = {0: "r", 1: "g", 2: "b"}
  for i, name in enumerate(reduced_data.keys()):
    axi = plt.subplot(2, 2, i+1)
    for label in range(num_labels):
      d = np.ravel(np.argwhere(Y == label))
      axi.scatter(
          reduced_data[name][d][:, 0], reduced_data[name][d][:, 1],
          c=color_mapping[label], edgecolors="none", alpha=0.6)
    axi.set_title(name)
  plt.show()
