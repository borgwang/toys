import time

import numpy as np
from common import gelu, layer_norm, linear, softmax
from generator import Generator
from transformers import GPT2Model, GPT2Tokenizer


class GPT2:

  hparams_dict = {
    "gpt2":        dict(n_layer=12, n_head=12, n_embed=768),   # 124M params
    "gpt2-medium": dict(n_layer=24, n_head=16, n_embed=1024),  # 350M params
    "gpt2-large":  dict(n_layer=36, n_head=20, n_embed=1280),  # 774M params
    "gpt2-xl":     dict(n_layer=48, n_head=25, n_embed=1600),  # 1558M params
  }


  def __init__(self, model_type, kv_cache=True):
    self.hparams = self.hparams_dict[model_type]
    self.ctx_size = 1024
    self.p = {k: v.numpy() for k, v in GPT2Model.from_pretrained(model_type).state_dict().items()}

    self.enable_kv_cache = kv_cache
    self.kv_cache = {}

  def forward(self, ids):
    wte, wpe = self.p["wte.weight"], self.p["wpe.weight"]
    x = wte[ids] + wpe[range(len(ids))]
    if self.enable_kv_cache and self.kv_cache:
      x = wte[ids[-1:]] + wpe[[len(ids)-1]]

    for i in range(self.hparams["n_layer"]):
      x = self.transformer_block(x, i)
    x = layer_norm(x, self.p["ln_f.weight"], self.p["ln_f.bias"])
    logits = x[-1] @ wte.T
    return logits

  def transformer_block(self, x, i):
    x = x + self.attn(layer_norm(x, self.p[f"h.{i}.ln_1.weight"], self.p[f"h.{i}.ln_1.bias"]), i)
    x = x + self.ffn(layer_norm(x, self.p[f"h.{i}.ln_2.weight"], self.p[f"h.{i}.ln_2.bias"]), i)
    return x

  def attn(self, x, i):
    T, C = x.shape
    n_head, hs = self.hparams["n_head"], C // self.hparams["n_head"]

    x = linear(x, self.p[f"h.{i}.attn.c_attn.weight"], self.p[f"h.{i}.attn.c_attn.bias"])
    q, k, v = [np.transpose(h.reshape((T, n_head, hs)), (1,0,2)) for h in np.split(x, 3, axis=-1)]

    if self.enable_kv_cache:
      if i in self.kv_cache:
        k = np.concatenate([self.kv_cache[i][0], k], axis=1)
        v = np.concatenate([self.kv_cache[i][1], v], axis=1)
      self.kv_cache[i] = (k, v)

    attn = softmax(q @ np.transpose(k, (0,2,1)) / hs**0.5 + (1 - np.tri(T, dtype=np.float32)) * -1e10)
    x = np.transpose(attn @ v, (1,0,2)).reshape((T, C))
    x = linear(x, self.p[f"h.{i}.attn.c_proj.weight"], self.p[f"h.{i}.attn.c_proj.bias"])
    return x

  def ffn(self, x, i):
    x = gelu(linear(x, self.p[f"h.{i}.mlp.c_fc.weight"], self.p[f"h.{i}.mlp.c_fc.bias"]))
    x = linear(x, self.p[f"h.{i}.mlp.c_proj.weight"], self.p[f"h.{i}.mlp.c_proj.bias"])
    return x


if __name__ == "__main__":
  model_type = "gpt2"
  model = GPT2(model_type, kv_cache=True)
  tokenizer = GPT2Tokenizer.from_pretrained(model_type)

  prompt = "Alan Turing theorized that computers would one day become"
  gen = Generator(model, tokenizer, t=0)

  st = time.monotonic()
  toks, text = gen(prompt)
  cost = time.monotonic() - st
  print(f"cost: {cost:.2f}s, {len(toks)/cost:.2f} tok/s")
