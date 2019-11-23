import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import ResNet, BasicBlock


class MnistResNet(ResNet):

    def __init__(self):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)

    """
        self.fc_tail = nn.Linear(50, 10)

    def forward(self, x, vae_feat):
        out = super().forward(x)
        out = torch.cat([out, vae_feat], 1)
        out = self.fc_tail(out)
        return out
    """


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


class VAE(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
