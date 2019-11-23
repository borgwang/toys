import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 84)
        self.fc2 = nn.Linear(84, 10)

        # weight initialization
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.dropout1(out)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

class Discriminator(nn.Module):

    def __init__(self, z_dim):
        super().__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.layers = nn.Sequential(
            nn.Linear(794, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 1),
            nn.Sigmoid())

    def forward(self, x, labels):
        z = x.view(x.size(0), 784)
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.layers(x)
        return out.squeeze()


class Generator(nn.Module):

    def __init__(self, z_dim):
        super().__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.z_dim = z_dim
        self.layers = nn.Sequential(
            nn.Linear(self.z_dim + 10, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 784),
            nn.Sigmoid())

    def forward(self, x, labels):
        z = x.view(x.size(0), self.z_dim)
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.layers(x)
        out = out.view(x.size(0), 28, 28)
        return out
