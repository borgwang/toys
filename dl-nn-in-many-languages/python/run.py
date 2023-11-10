import numpy as np
import pandas as pd

N_EPOCH = 1000
LR = 0.0003
BS = 30

def load_dataset(path):
  feat_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
  label_col = ["label"]
  data = pd.read_csv(path, names=feat_cols + label_col)
  X, labels = data[feat_cols].values, data[label_col].values.ravel()
  label_map = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
  labels = [label_map[label] for label in labels]
  # onehot
  Y = np.zeros((len(labels), 3))
  Y[np.arange(len(labels)), labels] = 1
  # split train/test
  random_idx = np.arange(len(X))
  np.random.shuffle(random_idx)
  X, Y = X[random_idx], Y[random_idx]
  train_size = int(len(X) * 0.8)
  trainX, trainY = X[:train_size], Y[:train_size]
  testX, testY = X[train_size:], Y[train_size:]
  # normalize
  mean = np.mean(trainX, 0)
  std = np.std(trainX, 0)
  trainX = (trainX - mean) / std
  testX = (testX - mean) / std
  return trainX, trainY, testX, testY

def softmax(inputs):
  return np.exp(inputs) / np.sum(np.exp(inputs), 1)[:, None]

def init_params(in_dim, out_dim, hidden_dim=20):
  bound1 = np.sqrt(6.0 / (in_dim + hidden_dim))
  W1 = np.random.uniform(-bound1, bound1, size=[in_dim, hidden_dim])
  b1 = np.zeros(hidden_dim)
  bound2 = np.sqrt(6.0 / (hidden_dim + out_dim))
  W2 = np.random.uniform(-bound2, bound2, size=[hidden_dim, out_dim])
  b2 = np.zeros(out_dim)
  return [W1, b1, W2, b2]

def train(batch_x, batch_y, params):
  # forward
  W1, b1, W2, b2 = params
  h1 = np.dot(batch_x, W1) + b1
  a1 = np.copy(h1)
  a1[a1 < 0.0] = 0.0
  h2 = np.dot(a1, W2) + b2
  p = softmax(h2)

  # NLL loss
  loss = np.mean(-np.log(np.sum(p * batch_y, 1)))
  print(f"loss: {loss:.4f}")

  # backward
  dl_dh2 = p - batch_y  # [batch, 3]
  dl_dW2 = np.dot(a1.T, dl_dh2)
  dl_db2 = np.sum(dl_dh2, 0)
  dl_da1 = np.dot(dl_dh2, W2.T)
  da1_dh1 = (h1 > 0).astype(float)
  dl_dh1 = dl_da1 * da1_dh1
  dl_dW1 = np.dot(batch_x.T, dl_dh1)
  dl_db1 = np.sum(dl_dh1, 0)

  grads = [dl_dW1, dl_db1, dl_dW2, dl_db2]
  for param, grad in zip(params, grads):
    param -= LR * grad

def evaluate(batch_x, batch_y, params):
  W1, b1, W2, b2 = params
  h1 = np.dot(batch_x, W1) + b1
  a1 = np.copy(h1)
  a1[a1 < 0.0] = 0.0
  h2 = np.dot(a1, W2) + b2
  p = softmax(h2)
  pred_labels = np.argmax(p, 1)
  true_labels = np.argmax(batch_y, 1)
  precision = np.mean(pred_labels == true_labels)
  print(f"precision: {precision:.4f}")

def main():
  # prepare dataset
  train_x, train_y, test_x, test_y = load_dataset( "../data/iris.data")
  params = init_params(4, 3)
  for epoch in range(N_EPOCH):
    for b in range(len(train_x) // BS):
      batch_x = train_x[b * BS: (b + 1) * BS]
      batch_y = train_y[b * BS: (b + 1) * BS]
      train(batch_x, batch_y, params)
    if epoch % 100 == 0:
      evaluate(test_x, test_y, params)


if __name__ == "__main__":
  main()
