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

def stream_printer(ids):
  print(tokenizer.decode(ids), end="", flush=True)

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

  def __init__(self, model_type):
    self.hparams = self.hparams_dict[model_type]
    self.p = {k: v.numpy() for k, v in GPT2Model.from_pretrained(model_type).state_dict().items()}

  def generate(self, start_ids, max_new_tokens, temperature=1.0, stream_printer=None):
    ret_p, ret_ids = [], []
    cnt = 0
    if stream_printer:
      stream_printer(start_ids)

    while cnt < max_new_tokens:
      ids_cond = (start_ids + ret_ids)[-self.context_len:]

      # autoregressive sampling
      p = self.forward(ids_cond, temperature, only_last=True)
      new_p, new_ids = [p], [sample(p)]

      ret_p += new_p
      ret_ids += new_ids
      cnt += len(new_ids)
      if stream_printer:
        stream_printer(new_ids)

    if stream_printer:
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

  def forward(self, ids, temperature=1.0, only_last=False):
    """minimal numpy implementation of gpt2 forward pass"""
    wte, wpe = self.p["wte.weight"], self.p["wpe.weight"]
    x = wte[ids] + wpe[range(len(ids))]
    for i in range(self.hparams["n_layer"]):
      x = self.transformer_block(x, i)
    x = layer_norm(x, self.p["ln_f.weight"], self.p["ln_f.bias"])
    x = x[-1] if only_last else x
    return softmax((x @ wte.T) / (temperature + 1e-8))


class GPT2KvCache(GPT2):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.kv_cache = {}

  def mha(self, x, i):
    T, C = x.shape
    x = linear(x, self.p[f"h.{i}.attn.c_attn.weight"], self.p[f"h.{i}.attn.c_attn.bias"])
    n_head, hs = self.hparams["n_head"], C // self.hparams["n_head"]
    q, k, v = [np.transpose(h.reshape((T, n_head, hs)), (1,0,2)) for h in np.split(x, 3, axis=-1)]
    if i in self.kv_cache:
      cache_attn = np.pad(self.kv_cache[i], ((0,0),(0,0),(0,1)))
      new_attn = softmax(q[:, -1:, :] @ np.transpose(k, (0,2,1)) / hs**0.5)
      attn = np.concatenate((cache_attn, new_attn), axis=1)
    else:
      attn = softmax(q @ np.transpose(k, (0,2,1)) * hs**(-0.5) + (1 - np.tri(T, dtype=np.float32)) * -1e10)
    self.kv_cache[i] = attn
    x = np.transpose(attn @ v, (1,0,2)).reshape((T, C))
    x = linear(x, self.p[f"h.{i}.attn.c_proj.weight"], self.p[f"h.{i}.attn.c_proj.bias"])
    return x

class GPT2SpeculativeSample(GPT2):

  def generate(self, start_ids, max_new_tokens, temperature=1.0, stream_printer=None, draft_model=None, K=4):
    ret_p, ret_ids = [], []
    cnt = 0
    if stream_printer:
      stream_printer(start_ids)

    while cnt < max_new_tokens:
      ids_cond = (start_ids + ret_ids)[-self.context_len:]

      assert draft_model is not None
      # 1. sample K steps from draft model
      p_draft, ids_draft = draft_model.generate(ids_cond, K, temperature=temperature)
      # 2. forward target model
      p = self.forward(ids_cond + ids_draft, temperature=temperature)[-K-1:]
      # 3. loop throught draft tokens and perform reject samping
      new_p, new_ids = [], []
      all_accepted = True
      for i in range(K):
        j = ids_draft[i]
        if np.random.uniform() >= min(1, p[i][j]/p_draft[i][j]):
          # if current draft token j is rejected, we resample a token from normalized max(0, p-q)
          new_ids.append(sample(np.maximum(p[i] - p_draft[i], 0)))
          new_p.append(p[i])
          all_accepted = False
          break
        new_ids.append(j)
        new_p.append(p[i])
      if all_accepted:
        # sample extra token x_{n+k+1} if all draft tokens were accepted
        new_ids.append(sample(p[-1]))
        new_p.append(p[-1])

      ret_p += new_p
      ret_ids += new_ids
      cnt += len(new_ids)
      if stream_printer:
        stream_printer(new_ids)

    if stream_printer:
      print()
    return np.vstack(ret_p), ret_ids


# configs
max_new_tokens = 100
target_model_name = "gpt2"

prompt = "Alan Turing theorized that computers would one day become"
tokenizer = GPT2Tokenizer.from_pretrained(target_model_name)
start_ids = tokenizer(prompt)["input_ids"]

#target_model = GPT2(target_model_name)
target_model = GPT2KvCache(target_model_name)
st = time.monotonic()
_, ids = target_model.generate(start_ids, max_new_tokens, temperature=0, stream_printer=stream_printer)
cost = time.monotonic() - st
print(f"cost: {cost:.2f}s, {len(ids)/cost:.2f} tokens/s")
