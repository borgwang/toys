import numpy as np

N_EXAMPLES = 150
IN_DIM = 4
OUT_DIM = 3
HIDDEN_DIM = 20

np.random.seed(31)

# generate split
split_idx = int(N_EXAMPLES * 0.8)
idx = np.arange(N_EXAMPLES)
np.random.shuffle(idx)
with open("./split.txt", "w") as f:
  f.write(",".join(map(str, idx[:split_idx])) + "\n")
  f.write(",".join(map(str, idx[split_idx:])) + "\n")

# generate init params
bound1 = np.sqrt(6.0 / (IN_DIM + HIDDEN_DIM))
W1 = np.random.uniform(-bound1, bound1, size=[IN_DIM, HIDDEN_DIM])
b1 = np.zeros(HIDDEN_DIM)
bound2 = np.sqrt(6.0 / (HIDDEN_DIM + OUT_DIM))
W2 = np.random.uniform(-bound2, bound2, size=[HIDDEN_DIM, OUT_DIM])
b2 = np.zeros(OUT_DIM)

# save as npz
np.savez("./params.npz", W1=W1, b1=b1, W2=W2, b2=b2)

# save as params text
with open("./params.txt", "w") as f:
  f.write(",".join(f"{p:.8f}" for p in W1.flatten()) + "\n")
  f.write(",".join(f"{p:.8f}" for p in b1) + "\n")
  f.write(",".join(f"{p:.8f}" for p in W2.flatten()) + "\n")
  f.write(",".join(f"{p:.8f}" for p in b2) + "\n")
