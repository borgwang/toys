import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from nets import Discriminator
from nets import Generator


def plot2img(tensors):
    img = tensors.detach().numpy()
    plt.figure()
    for i, x in enumerate(img):
        plt.subplot(len(img) // 10, 10, i+1)
        x_ = (x * 255.0).astype(int).reshape((28, 28))
        plt.imshow(x_, cmap="gray")
        plt.axis("off")
    plt.savefig("./gan_data.png")
    plt.close()


def gan_augment(x, y, seed, n_samples=None):
    if n_samples is None:
        n_samples = len(x)

    lr = 3e-4
    num_ep = 300
    z_dim = 100
    model_path = "./gan_checkpoint_%d.pth" % seed

    device = "cuda" if torch.cuda.is_available() else "cpu"
    G = Generator(z_dim).to(device)
    D = Discriminator(z_dim).to(device)
    bce_loss = nn.BCELoss()
    G_optim = optim.Adam(G.parameters(), lr=lr*3, betas=(0.5, 0.999))
    D_optim = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    batch = 64
    train_x = torch.Tensor(x)
    train_labels = torch.LongTensor(y)

    if os.path.exists(model_path):
        print("load trained GAN...")
        state = torch.load(model_path)
        G.load_state_dict(state["G"])
    else:
        print("training a new GAN...")
        for epoch in range(num_ep):
            for _ in range(len(train_x) // batch):
                idx = np.random.choice(range(len(train_x)), batch)
                batch_x = train_x[idx].to(device)
                batch_labels = train_labels[idx].to(device)

                y_real = torch.ones(batch).to(device)
                y_fake = torch.zeros(batch).to(device)

                # train D with real images
                D.zero_grad()
                D_real_out = D(batch_x, batch_labels).squeeze()
                D_real_loss = bce_loss(D_real_out, y_real)

                # train D with fake images
                z_ = torch.randn((batch, z_dim)).view(-1, z_dim, 1, 1).to(device)
                fake_labels = torch.randint(0, 10, (batch,)).to(device)
                G_out = G(z_, fake_labels)

                D_fake_out = D(G_out, fake_labels).squeeze()
                D_fake_loss = bce_loss(D_fake_out, y_fake)
                D_loss = D_real_loss + D_fake_loss
                D_loss.backward()
                D_optim.step()

                # train G
                G.zero_grad()
                z_ = torch.randn((batch, z_dim)).view(-1, z_dim, 1, 1).to(device)
                fake_labels = torch.randint(0, 10, (batch,)).to(device)
                G_out = G(z_, fake_labels)
                D_out = D(G_out, fake_labels).squeeze()
                G_loss = bce_loss(D_out, y_real)
                G_loss.backward()
                G_optim.step()

            plot2img(G_out[:50].cpu())
            print("epoch: %d G_loss: %.2f D_loss: %.2f" % 
                  (epoch, G_loss, D_loss))
        state = {"G": G.state_dict(), "D": D.state_dict()}
        torch.save(state, model_path)

    with torch.no_grad():
        z_ = torch.randn((n_samples, z_dim)).view(-1, z_dim, 1, 1).to(device)
        fake_labels = torch.randint(0, 10, (n_samples,)).to(device)
        G_samples = G(z_, fake_labels)
        samples = G_samples.cpu().numpy().reshape((-1, 28, 28, 1))
    return samples, fake_labels.cpu().numpy()
