import argparse

import Augmentor
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gan_augment import gan_augment
from nets import MnistResNet, SiameseNet
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from utils import make_dataset


def plot_samples(samples, path):
  plt.figure()
  for i, x in enumerate(samples):
    plt.subplot(len(samples) // 10, 10, i+1)
    x_ = (x * 255).astype(int).reshape((28, 28))
    plt.imshow(x_, cmap="gray")
    plt.axis("off")
  plt.savefig(path)
  plt.close()


def statistical_ml(dataset):
  """classic ML algorithms"""

  def fit_and_evaluate(model):
    train_x, train_y, test_x, test_y = dataset
    model.fit(train_x, train_y)
    test_pred = model.predict(test_x)
    print("accuracy: %.4f" % accuracy_score(test_y, test_pred))

  # statistical ML models
  print("\nRandomForestClassifier")
  rf = RandomForestClassifier(n_estimators=200, max_features=10)
  fit_and_evaluate(rf)

  print("\nLogisticRegression")
  lr = LogisticRegression(solver="lbfgs", multi_class="auto")
  fit_and_evaluate(lr)

  print("\nGradientBoostingClassifier")
  gbdt = GradientBoostingClassifier()
  fit_and_evaluate(gbdt)

  print("\nSVMClassifier")
  svc = SVC()
  fit_and_evaluate(svc)

  print("\nKNeighborsClassifier")
  knn = KNeighborsClassifier(n_neighbors=1)
  fit_and_evaluate(knn)


def deep_learning(dataset):

  def preprocess(dataset, data_augmentation=True):
    train_x, train_y, test_x, test_y = dataset
    train_x = train_x.reshape((-1, 28, 28, 1))
    test_x = test_x.reshape((-1, 28, 28, 1))
    plot_samples(train_x, "./origin-%d.png" % args.seed)

    if data_augmentation:
      n_samples = 1024
      # data augmentation
      p = Augmentor.Pipeline()
      p.set_seed(args.seed)
      p.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=10)
      p.random_distortion(probability=0.8, grid_width=3, grid_height=3, magnitude=2)
      p.skew(probability=0.8, magnitude=0.3)
      p.shear(probability=0.5, max_shear_left=3, max_shear_right=3)
      generator = p.keras_generator_from_array(train_x, train_y, n_samples, scaled=False)

      train_x, train_y = next(generator)
      train_x = np.clip(train_x, 0, 1)
      plot_samples(train_x[:50], "./data_augmentation.png")

    if args.gan_ratio > 0:
      n_samples = int(args.gan_ratio * len(train_x))
      a_x, a_y = gan_augment(train_x, train_y, args.seed, n_samples)
      train_x = np.concatenate([train_x, a_x])
      train_y = np.concatenate([train_y, a_y])

    # convert to NCWH tensor
    train_x = np.transpose(train_x, (0, 3, 1, 2))
    test_x = np.transpose(test_x, (0, 3, 1, 2))
    train_x = torch.Tensor(train_x)
    test_x = torch.Tensor(test_x)
    train_y = torch.LongTensor(train_y)
    test_y = torch.LongTensor(test_y)
    return train_x, train_y, test_x, test_y

  def fit_and_evaluate(dataset, net):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(device)
    train_x, train_y, test_x, test_y = dataset

    # resize image to (224, 224, 1) to fit resnet
    train_x = F.interpolate(train_x, size=224)
    test_x = F.interpolate(test_x, size=224)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.001)
    batch = 64
    running_loss = 0
    for epoch in range(args.num_ep):
      for i in range(len(train_x) // batch):
        idx = np.random.choice(range(len(train_x)), batch)
        x, y = train_x[idx].to(device), train_y[idx].to(device)
        optimizer.zero_grad()
        out = net(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        running_loss = 0.99 * running_loss + 0.01 * train_loss
      if epoch % 20 == 0:
        print(running_loss)

    batch = 100
    preds = []
    with torch.no_grad():
      for i in range(0, len(test_x), batch):
        x = test_x[i: i+batch].to(device)
        out = net(x)
        preds.extend(out.argmax(1).cpu().numpy().tolist())
    print("accuracy: %.4f" % accuracy_score(test_y.cpu().numpy().tolist(), preds))

  dataset = preprocess(dataset, data_augmentation=True)
  fit_and_evaluate(dataset, MnistResNet())
  # fit_and_evaluate(dataset, LeNet())


def siamese_net(dataset):
  """buggy"""
  def preprocess(dataset, data_augmentation=True):
    train_x, train_y, test_x, test_y = dataset
    train_x = train_x.reshape((-1, 28, 28, 1))
    test_x = test_x.reshape((-1, 28, 28, 1))

    if data_augmentation:
      n_samples = 1024
      # data augmentation with Augmentor
      p = Augmentor.Pipeline()
      p.set_seed(args.seed)
      p.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=10)
      p.random_distortion(probability=0.8, grid_width=3, grid_height=3, magnitude=2)
      p.skew(probability=0.8, magnitude=0.3)
      p.shear(probability=0.5, max_shear_left=3, max_shear_right=3)
      generator = p.keras_generator_from_array(train_x, train_y, n_samples, scaled=False)

      origin_x, origin_y = train_x, train_y
      train_x, train_y = next(generator)
      train_x = np.clip(train_x, 0, 1)

    # convert to tensr
    train_x = torch.Tensor(train_x.reshape((-1, 784)))
    origin_x = torch.Tensor(origin_x.reshape((-1, 784)))
    test_x = torch.Tensor(test_x.reshape(-1, 784))
    train_y = torch.LongTensor(train_y)
    origin_y = torch.LongTensor(origin_y)
    test_y = torch.LongTensor(test_y)
    return train_x, train_y, test_x, test_y, origin_x, origin_y

  def fit_and_evaluate(dataset, net):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(device)

    train_x, train_y, test_x, test_y, origin_x, origin_y = dataset
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    batch = 128
    running_loss = 0
    for epoch in range(args.num_ep):
      for i in range(len(train_x) // batch):
        idx = np.random.choice(range(len(train_x)), batch * 2)
        x, y = train_x[idx].to(device), train_y[idx].to(device)
        x1, x2 = torch.split(x, batch, dim=0)
        y1, y2 = torch.split(y, batch, dim=0)
        y = (y1 == y2).float()

        optimizer.zero_grad()
        out = net(x1, x2)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        running_loss = 0.99 * running_loss + 0.01 * train_loss
      if epoch % 20 == 0:
        print(running_loss)

    batch = 1000
    preds = []
    with torch.no_grad():
      x_, _ = origin_x.to(device), origin_y.to(device)
      for i in range(0, len(test_x), batch):
        x = test_x[i: i+batch].to(device)
        out = net.predict(x, x_)
        preds.extend(out.argmax(1).cpu().numpy().tolist())
    print("accuracy: %.4f" % accuracy_score(test_y.cpu().numpy().tolist(), preds))

  dataset = preprocess(dataset, data_augmentation=True)
  fit_and_evaluate(dataset, SiameseNet())


def main():
  # seed
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)
  torch.backends.cudnn.deterministic = True
  # dataset
  dataset = make_dataset()
  # statistical_ml(dataset)
  deep_learning(dataset)
  # siamese_net(dataset)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, default=31)
  parser.add_argument("--lr", type=float, default=3e-3)
  parser.add_argument("--num_ep", type=int, default=301)
  parser.add_argument("--gan_ratio", type=float, default=0)
  args = parser.parse_args()
  main()
