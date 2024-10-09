import time
from typing import List, Tuple

import numpy as np

N_EPOCH = 1000
LR = 0.0003
BS = 30
IN_DIM = 4
OUT_DIM = 3
HIDDEN_DIM = 20
LR = 0.0003

def load_dataset(path:str, split_path:str):
  # Read CSV without pandas
  with open(path, "r") as f:
    rows = list(map(lambda x: x.strip().split(","), f.readlines()))
  X = np.array([[float(val) for val in row[:IN_DIM]] for row in rows], dtype=np.float32)
  # normalize feats
  X = (X - np.mean(X, 0)) / np.std(X, 0)
  # onehot labels
  label_map = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
  labels = [label_map[row[IN_DIM]] for row in rows]
  Y = np.zeros((len(labels), 3), dtype=np.float32)
  Y[np.arange(len(labels)), labels] = 1
  # split train/test
  with open(split_path, "r") as f:
    train_idx, test_idx = map(lambda x: list(map(int, x.strip().split(","))), f.readlines())
  return X[train_idx], Y[train_idx], X[test_idx], Y[test_idx]

def softmax(inputs):
  return np.exp(inputs) / np.sum(np.exp(inputs), 1)[:, None]

def train_step(params:List[np.ndarray], batch_x:np.ndarray, batch_y:np.ndarray):
  # forward
  W1, b1, W2, b2 = params

  h1 = np.dot(batch_x, W1) + b1
  a1 = np.copy(h1)
  a1[a1 < 0.0] = 0.0
  h2 = np.dot(a1, W2) + b2

  p = softmax(h2)

  # loss
  # loss = np.mean(-np.log(np.sum(p * batch_y, 1)))
  # print(f"loss: {loss:.4f}")

  # backward
  dl_dh2 = p - batch_y
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

def evaluate(params, batch_x, batch_y) -> Tuple[float, float]:
  W1, b1, W2, b2 = params
  h1 = np.dot(batch_x, W1) + b1
  a1 = np.copy(h1)
  a1[a1 < 0.0] = 0.0
  h2 = np.dot(a1, W2) + b2
  p = softmax(h2)
  pred_labels = np.argmax(p, 1)
  true_labels = np.argmax(batch_y, 1)
  precision = np.mean(pred_labels == true_labels)
  loss = np.mean(-np.log(np.sum(p * batch_y, 1)))
  return precision, loss

def main():
  # timing
  st = time.perf_counter()

  # prepare dataset
  train_x, train_y, test_x, test_y = load_dataset("../data/iris.data", "../data/split.txt")
  # load params
  with open("../data/params.txt", "r") as f:
    W1 = np.array(list(map(float, f.readline().split(","))), dtype=np.float32).reshape([IN_DIM, HIDDEN_DIM])
    b1 = np.array(list(map(float, f.readline().split(","))), dtype=np.float32).reshape([HIDDEN_DIM])
    W2 = np.array(list(map(float, f.readline().split(","))), dtype=np.float32).reshape([HIDDEN_DIM, OUT_DIM])
    b2 = np.array(list(map(float, f.readline().split(","))), dtype=np.float32).reshape([OUT_DIM])
  params = [W1, b1, W2, b2]
  for _ in range(N_EPOCH):
    for b in range(len(train_x) // BS):
      batch_x = train_x[b * BS: (b + 1) * BS]
      batch_y = train_y[b * BS: (b + 1) * BS]
      train_step(params, batch_x, batch_y)
  precision, loss = evaluate(params, test_x, test_y)
  print(f"precision={precision:.6f}, loss={loss:.6f}")

  # timing
  et = time.perf_counter()
  print(f"time={1000*(et - st):.4f} ms")


if __name__ == "__main__":
  main()
