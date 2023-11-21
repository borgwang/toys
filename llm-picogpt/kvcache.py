import time

import numpy as np
from transformers import GPT2Model, GPT2Tokenizer


def layer_norm(x, w, b, eps=1e-5):
  mean = np.mean(x, axis=-1, keepdims=True)
  var = np.var(x, axis=-1, keepdims=True)
  return ((x - mean) / (var + eps)**0.5) * w + b

def softmax(x, axis=-1):
  x -= x.max(axis=axis, keepdims=True)
  x = np.exp(x, x)
  x /= x.sum(axis=axis, keepdims=True)
  return x

def linear(x, w, b):
  return x @ w + b

def gelu(x):
  return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def sample(p):
  if np.sum(p) != 1:
    p = p / p.sum()
  return np.random.choice(range(len(p)), p=p)

class GPT2:

  hparams_dict = {
    "gpt2":        dict(n_layer=12, n_head=12, n_embed=768),   # 124M params
    "gpt2-medium": dict(n_layer=24, n_head=16, n_embed=1024),  # 350M params
    "gpt2-large":  dict(n_layer=36, n_head=20, n_embed=1280),  # 774M params
    "gpt2-xl":     dict(n_layer=48, n_head=25, n_embed=1600),  # 1558M params
  }

  context_len = 1024

  def __init__(self, model_type, temperature=1.0):
    self.tokenizer = GPT2Tokenizer.from_pretrained(model_type)
    self.hparams = self.hparams_dict[model_type]
    self.p = {k: v.numpy() for k, v in GPT2Model.from_pretrained(model_type).state_dict().items()}
    self.t = temperature
    self.stream = True

  def generate(self, inputs, max_new_tokens):
    start_ids = self.tokenizer(inputs)["input_ids"]

    ret_p, ret_ids = [], []
    cnt = 0
    if self.stream:
      self.streamprint(start_ids)

    while cnt < max_new_tokens:
      ids_cond = (start_ids + ret_ids)[-self.context_len:]

      # autoregressive sampling
      p = self.forward(ids_cond, only_last=True)
      new_p, new_ids = [p], [sample(p)]

      ret_p += new_p
      ret_ids += new_ids
      cnt += len(new_ids)
      if self.stream:
        self.streamprint(new_ids)

    if self.stream:
      print()
    return np.vstack(ret_p), ret_ids

  def mlp(self, x, i):
    x = gelu(linear(x, self.p[f"h.{i}.mlp.c_fc.weight"], self.p[f"h.{i}.mlp.c_fc.bias"]))
    x = linear(x, self.p[f"h.{i}.mlp.c_proj.weight"], self.p[f"h.{i}.mlp.c_proj.bias"])
    return x

  def mha(self, x, i):
    T, C = x.shape
    x = linear(x, self.p[f"h.{i}.attn.c_attn.weight"], self.p[f"h.{i}.attn.c_attn.bias"])
    n_head, hs = self.hparams["n_head"], C // self.hparams["n_head"]
    q, k, v = [np.transpose(h.reshape((T, n_head, hs)), (1,0,2)) for h in np.split(x, 3, axis=-1)]
    attn = softmax(q @ np.transpose(k, (0,2,1)) / hs**0.5 + (1 - np.tri(T, dtype=np.float32)) * -1e10)
    x = np.transpose(attn @ v, (1,0,2)).reshape((T, C))
    x = linear(x, self.p[f"h.{i}.attn.c_proj.weight"], self.p[f"h.{i}.attn.c_proj.bias"])
    return x

  def transformer_block(self, x, i):
    x = x + self.mha(layer_norm(x, self.p[f"h.{i}.ln_1.weight"], self.p[f"h.{i}.ln_1.bias"]), i)
    x = x + self.mlp(layer_norm(x, self.p[f"h.{i}.ln_2.weight"], self.p[f"h.{i}.ln_2.bias"]), i)
    return x

  def forward(self, ids, only_last=False):
    """minimal numpy implementation of gpt2 forward pass"""
    wte, wpe = self.p["wte.weight"], self.p["wpe.weight"]
    x = wte[ids] + wpe[range(len(ids))]
    for i in range(self.hparams["n_layer"]):
      x = self.transformer_block(x, i)
    x = layer_norm(x, self.p["ln_f.weight"], self.p["ln_f.bias"])
    x = x[-1] if only_last else x
    return softmax((x @ wte.T) / (self.t + 1e-8))

  def streamprint(self, ids):
    print(self.tokenizer.decode(ids), end="", flush=True)


class GPT2KVCache(GPT2):

  def __init__(self, model_type, temperature=1.0):
    super().__init__(model_type, temperature)
    self.cache = {}

  def mha(self, x, i):
    T, C = x.shape
    x = linear(x, self.p[f"h.{i}.attn.c_attn.weight"], self.p[f"h.{i}.attn.c_attn.bias"])
    n_head, hs = self.hparams["n_head"], C // self.hparams["n_head"]
    q, k, v = [np.transpose(h.reshape((T, n_head, hs)), (1,0,2)) for h in np.split(x, 3, axis=-1)]
    if i in self.cache:
      k = np.concatenate([self.cache[i][0], k], axis=1)
      v = np.concatenate([self.cache[i][1], v], axis=1)
    self.cache[i] = (k, v)

    attn = softmax(q @ np.transpose(k, (0,2,1)) / hs**0.5 + (1 - np.tri(T, dtype=np.float32)) * -1e10)
    x = np.transpose(attn @ v, (1,0,2)).reshape((T, C))
    x = linear(x, self.p[f"h.{i}.attn.c_proj.weight"], self.p[f"h.{i}.attn.c_proj.bias"])
    return x

  def forward(self, ids, only_last=False):
    wte, wpe = self.p["wte.weight"], self.p["wpe.weight"]
    if not self.cache:
      x = wte[ids] + wpe[range(len(ids))]
    else:
      x = wte[ids[-1:]] + wpe[[len(ids)-1]]

    for i in range(self.hparams["n_layer"]):
      x = self.transformer_block(x, i)
    x = layer_norm(x, self.p["ln_f.weight"], self.p["ln_f.bias"])
    x = x[-1] if only_last else x
    #print(f"\ncache_size={self.get_cache_size()/1024/1024:.2f} MB. #tokens={len(ids)}")
    return softmax((x @ wte.T) / (self.t + 1e-8))


if __name__ == "__main__":
  model_type = "gpt2"
  max_new_tokens = 100
  prompt = "Alan Turing theorized that computers would one day become"

  print("without kv caching")
  target_model = GPT2(model_type, temperature=0)
  st = time.monotonic()
  _, ids = target_model.generate(prompt, max_new_tokens)
  cost = time.monotonic() - st
  print(f"cost: {cost:.2f}s, {len(ids)/cost:.2f} tokens/s")

  print("with kv caching")
  target_model = GPT2KVCache(model_type, temperature=0)
  st = time.monotonic()
  _, ids = target_model.generate(prompt, max_new_tokens)
  cost = time.monotonic() - st
  print(f"cost: {cost:.2f}s, {len(ids)/cost:.2f} tokens/s")
