import os   

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dl_models import VAE


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def vae_augment(x):
    lr = 3e-3
    batch = 64
    num_ep = 300
    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = VAE().to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    model_path = "./vae_checkpoint.pth"

    train_x = torch.Tensor(x.reshape((x.shape[0], -1)))
    if os.path.exists(model_path):
        print("load trained VAE...")
        state = torch.load(model_path)
        net.load_state_dict(state["net"])
    else:
        print("training a new VAE...")
        for epoch in range(num_ep):
            train_loss = 0
            for _ in range(len(train_x) // batch):
                batch_idx = np.random.choice(range(len(train_x)), batch)
                batch_x = train_x[batch_idx].to(device)

                optimizer.zero_grad()
                recon_batch, mu, logvar = net(batch_x)
                loss = loss_function(recon_batch, batch_x, mu, logvar)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            print(train_loss)
        state = {"net": net.state_dict()}
        torch.save(state, model_path)

    with torch.no_grad():
        _, mu, logvar = net(train_x.to(device))
        feat = torch.cat([mu, logvar], 1)
    return feat.cpu()
