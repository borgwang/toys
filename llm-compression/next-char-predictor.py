import numpy as np
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

with open("simple.txt", "r") as f:
  msg = f.readline().strip()

vocab = set(msg) | set([""])
vocab_size = len(vocab)
c2i = dict(zip(vocab, range(vocab_size)))
i2c = {i: c for c, i in c2i.items()}

# make training data
data = [[c2i[msg[i]], c2i[msg[i+1]]] for i in range(0, len(msg) - 1)]
data += [[c2i[""], c2i[msg[0]]]]
data = np.array(data)
emb_size = 32

class Layer(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(emb_size, emb_size)
    self.fc2 = nn.Linear(emb_size, emb_size)
    self.gelu = nn.GELU()

  def forward(self, x):
    return self.fc2(self.gelu(self.fc1(x)))

class Predictor(nn.Module):
  def __init__(self):
    super().__init__()
    self.cte = nn.Embedding(vocab_size, emb_size)
    self.layers = nn.ModuleList([Layer() for _ in range(4)])
    self.proj = nn.Linear(emb_size, vocab_size)

  def forward(self, x, y=None):
    x = self.cte(x)
    for layer in self.layers:
      x = layer(x)
    logits = self.proj(x)
    loss = None
    if y is not None:
      loss = F.cross_entropy(logits, y)
    return logits, loss

model = Predictor()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
optimizer.zero_grad(set_to_none=True)

batch_size = 8
for i in tqdm.tqdm(range(10000)):
  batch = data[np.random.randint(0, len(data), batch_size)]
  x = torch.tensor(batch[:, 0], dtype=torch.int64)
  y = torch.tensor(batch[:, 1], dtype=torch.int64)
  logits, loss = model(x, y)
  loss.backward()
  optimizer.step()
  optimizer.zero_grad(set_to_none=True)

inp = ["", "a", "b", "c", "d"]
print(inp)
inp = [c2i[i] for i in inp]
model.eval()
with torch.no_grad():
  logits, _ = model(torch.tensor(inp))
  logits = logits.numpy()
  x = np.argmax(logits, 1)
  print([i2c[i] for i in x])

