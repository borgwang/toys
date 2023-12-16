import time

import numpy as np
import torch
from generator import Generator
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import gelu, layer_norm, linear, softmax


class Phi:

  model_dict = {"microsoft/phi-1", "microsoft/phi-1_5", "microsoft/phi-2"}

  def __init__(self, model_type, freq_base=10000, freq_scale=1.0):
    ptmodel = AutoModelForCausalLM.from_pretrained(model_type, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True)
    self.hparams = ptmodel.config.to_dict()
    self.ctx_size = self.hparams["n_positions"]
    self.p = {}
    for k, v in ptmodel.state_dict().items():
      k = k[12:] if k.startswith("transformer") else k
      v = v.numpy() if "embd" in k or "ln" in k or "bias" in k else v.T.contiguous().numpy()
      self.p[k] = v

    # precompute for RoPE
    dim = min(self.hparams["rotary_dim"], self.hparams["n_embd"] // self.hparams["n_head"])
    inv_freq = 1 / freq_base ** (np.arange(0, dim, 2)[:dim//2] / dim)
    t = freq_scale * np.outer(np.arange(self.ctx_size), inv_freq).astype(np.float32)
    self.cos, self.sin = np.cos(t), np.sin(t)

  def forward(self, ids, only_last=True):
    x = self.p["embd.wte.weight"][ids]
    for i in range(self.hparams["n_layer"]):
      x = self.transformer(x, i)
    x = layer_norm(x, self.p["lm_head.ln.weight"], self.p["lm_head.ln.bias"])
    x = x[-1] if only_last else x
    x = linear(x, self.p["lm_head.linear.weight"], self.p["lm_head.linear.bias"])
    return x

  def transformer(self, x, i):
    # parallel attention and feed-forward per GPT-J
    norm_x = layer_norm(x, self.p[f"h.{i}.ln.weight"], self.p[f"h.{i}.ln.bias"])
    return x + self.attn(norm_x, i) + self.ffn(norm_x, i)

  def attn(self, x, i):
    T, C = x.shape
    n_head, hs = self.hparams["n_head"], C // self.hparams["n_head"]
    x = linear(x, self.p[f"h.{i}.mixer.Wqkv.weight"], self.p[f"h.{i}.mixer.Wqkv.bias"])
    q, k, v = [h.reshape((T, n_head, hs)) for h in np.split(x, 3, axis=-1)]

    cos, sin = self.cos[:len(k)], self.sin[:len(k)]
    q, k = self.apply_rotation(q, cos[-len(q):], sin[-len(q):]), self.apply_rotation(k, cos, sin)

    q, k, v = (np.transpose(h, (1,0,2)) for h in (q, k, v))  # (n_head, T, hs)
    attn = softmax(q @ np.transpose(k, (0,2,1)) / hs**0.5 + (1 - np.tri(T, dtype=np.float32)) * -1e10)
    x = np.transpose(attn @ v, (1,0,2)).reshape((T, C))
    x = linear(x, self.p[f"h.{i}.mixer.out_proj.weight"], self.p[f"h.{i}.mixer.out_proj.bias"])
    return x

  def ffn(self, x, i):
    x = gelu(linear(x, self.p[f"h.{i}.mlp.fc1.weight"], self.p[f"h.{i}.mlp.fc1.bias"]))
    x = linear(x, self.p[f"h.{i}.mlp.fc2.weight"], self.p[f"h.{i}.mlp.fc2.bias"])
    return x

  @staticmethod
  def apply_rotation(x, cos, sin):
    # NOTE: use partial RoPE. only rotary the first hparams["rotaty_dim"] dim and keep the rest unchanged
    rotary_dim = cos.shape[-1] * 2
    x, x_pass = x[:,:,:rotary_dim], x[:,:,rotary_dim:]

    T, n_heads, hs = x.shape
    # split to real part and imaginary part
    x = x.reshape((T, n_heads, 2, hs//2))   # slightly different from RoPE from llama2
    x_r, x_i = x[:,:,0,:], x[:,:,1,:]
    # apply rotary transformation
    cos, sin = cos[:,None,:], sin[:,None,:]
    x_r_o, x_i_o = x_r*cos - x_i*sin, x_i*cos + x_r*sin
    # stack and reshape
    x_o = np.stack([x_r_o, x_i_o], axis=-1).reshape((T, n_heads, hs))
    # concat with the unchanged part
    return np.concatenate([x_o, x_pass], axis=-1)


if __name__ == "__main__":
  model_type = "microsoft/phi-1"
  prompt = """def hello_world():"""

  model = Phi(model_type)
  tokenizer = AutoTokenizer.from_pretrained(model_type, trust_remote_code=True)
  gen = Generator(model, tokenizer, t=0, max_tokens=100)

  st = time.monotonic()
  toks, text = gen(prompt)
  cost = time.monotonic() - st
  print(f"cost: {cost:.2f}s, {len(toks)/cost:.2f} tok/s")
