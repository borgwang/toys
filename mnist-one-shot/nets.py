import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18


class SiameseNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.head1 = nn.Linear(784, 128)
        self.head2 = nn.Linear(784, 128)
        self.body = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.75),
            nn.Linear(256, 128))
        self.tail = nn.Linear(128, 1)

    def forward(self, x1, x2):
        out1 = self.body(F.relu(self.head1(x1)))
        out2 = self.body(F.relu(self.head2(x2)))
        distance = torch.abs(out1 - out2)
        out = torch.sigmoid(self.tail(distance))
        return out

    def predict(self, x1, x_):
        probs = [self.forward(x1, x2) for x2 in x_]
        return torch.cat(probs, 1)


class MnistResNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(1, 3, kernel_size=1)
        self.resnet = resnet18(pretrained=True)

    def forward(self, x):
        out = self.conv0(x)
        out = self.resnet(out)
        return out


class LeNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(16)
        self.dp1 = nn.Dropout(0.5)
        self.dp2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 84)
        self.fc2 = nn.Linear(84, 10)

        # weight initialization
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.dp1(out)
        out = F.relu(self.fc1(out))
        out = self.dp2(out)
        out = self.fc2(out)
        return out


class Discriminator(nn.Module):

    def __init__(self, z_dim):
        super().__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.layers = nn.Sequential(
            nn.Linear(794, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
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
            nn.Linear(128, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid())

    def forward(self, x, labels):
        z = x.view(x.size(0), self.z_dim)
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.layers(x)
        out = out.view(x.size(0), 28, 28)
        return out
