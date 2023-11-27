import numpy as np
import requests
import tqdm


def layer_norm(x, w, b, eps=1e-5):
  mean = np.mean(x, axis=-1, keepdims=True)
  var = np.var(x, axis=-1, keepdims=True)
  return ((x - mean) / (var + eps)**0.5) * w + b

def rms_norm(x, w, eps=1e-5):
  return (x / ((x**2).mean(-1, keepdims=True) + eps)**0.5) * w

def gelu(x):
  return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def silu(x):
  return x * 1.0 / (1.0 + np.exp(-x))

def linear(x, w, b=None):
  x = x @ w
  if b is not None:
    x += b
  return x

def softmax(x, axis=-1):
  x -= x.max(axis=axis, keepdims=True)
  x = np.exp(x)
  x /= x.sum(axis=axis, keepdims=True)
  return x

def log_softmax(x, axis=-1):
  x -= x.max(axis=axis, keepdims=True)
  e_sum = np.exp(x).sum(axis=axis, keepdims=True)
  return x - np.log(e_sum)

def sample(p):
  if np.sum(p) != 1:
    p = p / p.sum()
  return int(np.random.choice(range(len(p)), p=p))

def download_file(url: str, fname: str, chunk_size=1024):
  """Helper function to download a file from a given url"""
  resp = requests.get(url, stream=True, timeout=60)
  total = int(resp.headers.get("content-length", 0))
  with open(fname, "wb") as file, tqdm.tqdm(
    desc=fname,
    total=total,
    unit="iB",
    unit_scale=True,
    unit_divisor=1024,
  ) as bar:
    for data in resp.iter_content(chunk_size=chunk_size):
      size = file.write(data)
      bar.update(size)

def onehot(labels, n_classes):
  return np.eye(n_classes, dtype=np.float32)[np.array(labels).reshape(-1)]
