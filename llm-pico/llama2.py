import os
import time

import numpy as np
import torch
from common import download_file, linear, rms_norm, silu, softmax
from generator import Generator
from sentencepiece import SentencePieceProcessor


class Tokenizer:
  # see https://github.com/facebookresearch/llama/blob/main/llama/tokenizer.py
  def __init__(self):
    self.sp_model = SentencePieceProcessor(model_file="./llama2-tokenizer.model")
    self.n_words:int = self.sp_model.vocab_size()
    self.bos_id:int = self.sp_model.bos_id()
    self.eos_id:int = self.sp_model.eos_id()

  def encode(self, s, bos=True, eos=False):
    t = self.sp_model.encode(s)
    if bos:
      t = [self.bos_id] + t
    if eos:
      t = t + [self.eos_id]
    return t

  def decode(self, t) -> str:
    return self.sp_model.decode(t)


class LLaMA2:
  model_dict = {"15M", "42M", "110M"}

  def __init__(self, model_type, kv_cache=True, freq_base=10000, freq_scale=1.0):
    assert model_type in self.model_dict
    state_dict = self.load_state_dict(model_type)
    self.hparams = state_dict["model_args"]
    self.ctx_size = self.hparams["max_seq_len"]
    self.p = {k: v.numpy() if "tok_embeddings" in k or "norm" in k else v.T.contiguous().numpy() for k, v in state_dict["model"].items()}

    # precompute for RoPE
    dim = self.hparams["dim"] // self.hparams["n_heads"]
    inv_freq = 1 / freq_base ** (np.arange(0, dim, 2)[:dim//2] / dim)
    t = freq_scale * np.outer(np.arange(self.ctx_size*4), inv_freq).astype(np.float32)
    self.cos, self.sin = np.cos(t), np.sin(t)

    # kv cache
    self.enable_kv_cache = kv_cache
    self.kv_cache = {}

  def forward(self, ids, only_last=True):
    x = self.p["tok_embeddings.weight"][ids]
    if self.enable_kv_cache and self.kv_cache:
      x = self.p["tok_embeddings.weight"][ids[-1:]]
    for i in range(self.hparams["n_layers"]):
      x = self.transformer_block(x, i)
    x = rms_norm(x, self.p["norm.weight"])
    x = x[-1] if only_last else x
    return x @ self.p["output.weight"]

  def transformer_block(self, x, i):
    x = x + self.attn(rms_norm(x, self.p[f"layers.{i}.attention_norm.weight"]), i)
    x = x + self.ffn(rms_norm(x, self.p[f"layers.{i}.ffn_norm.weight"]), i)
    return x

  def attn(self, x, i):
    T, C = x.shape
    n_head, hs = self.hparams["n_heads"], C // self.hparams["n_heads"]

    q, k, v = (linear(x, self.p[f"layers.{i}.attention.w{t}.weight"]).reshape((T, n_head, hs)) for t in "qkv")
    if self.enable_kv_cache:
      # cache kv before applying rotation
      if i in self.kv_cache:
        k = np.concatenate([self.kv_cache[i][0], k], axis=0)
        v = np.concatenate([self.kv_cache[i][1], v], axis=0)
      self.kv_cache[i] = (k, v)

    cos, sin = self.cos[:len(k)], self.sin[:len(k)]
    q = self.apply_rotation(q, cos[-len(q):], sin[-len(q):])
    k = self.apply_rotation(k, cos, sin)
    q, k, v = (np.transpose(h, (1,0,2)) for h in (q, k, v))  # (n_head, T, hs)

    attn = softmax(q @ np.transpose(k, (0,2,1)) / hs**0.5 + (1 - np.tri(T, dtype=np.float32)) * -1e10)
    x = np.transpose(attn @ v, (1,0,2)).reshape((T, C))
    x = linear(x, self.p[f"layers.{i}.attention.wo.weight"])
    return x

  def ffn(self, x, i):
    # see https://arxiv.org/pdf/2002.05202.pdf
    x = silu(linear(x, self.p[f"layers.{i}.feed_forward.w1.weight"])) * linear(x, self.p[f"layers.{i}.feed_forward.w3.weight"])
    x = linear(x, self.p[f"layers.{i}.feed_forward.w2.weight"])
    return x

  @staticmethod
  def apply_rotation(x, cos, sin):
    T, n_heads, hs = x.shape
    # split to real part and imaginary part
    x = x.reshape((T, n_heads, hs//2, 2))
    x_r, x_i = x[:,:,:,0], x[:,:,:,1]
    # apply rotary transformation
    cos, sin = cos[:,None,:], sin[:,None,:]
    x_r_o, x_i_o = x_r*cos - x_i*sin, x_i*cos + x_r*sin
    # stack and reshape
    x_o = np.stack([x_r_o, x_i_o], axis=-1).reshape((T, n_heads, hs))
    return x_o

  @staticmethod
  def load_state_dict(model_type):
    local_path = os.path.join("/tmp/pico-llama2", model_type + ".pt")
    if not os.path.exists(local_path):
      url = f"https://huggingface.co/karpathy/tinyllamas/resolve/main/stories{model_type}.pt"
      download_file(url, local_path)
    return torch.load(local_path)


if __name__ == "__main__":
  tmpdir = "/tmp/pico-llama2"
  if not os.path.exists(tmpdir):
    os.makedirs(tmpdir, exist_ok=True)

  model_type = "15M"
  model = LLaMA2(model_type, kv_cache=True)
  #model = LLaMA2(model_type, kv_cache=True, freq_scale=0.25)
  tokenizer = Tokenizer()

  prompt = "Lily is a bad girl"
  gen = Generator(model, tokenizer, t=0)

  st = time.monotonic()
  toks, text = gen(prompt)
  cost = time.monotonic() - st
  print(f"cost: {cost:.2f}s, {len(toks)/cost:.2f} tok/s")
