"""
Load weights of gpt2-* and sample text using numpy (float32).

- refs
  - https://github.com/jaymody/picoGPT
  - https://github.com/karpathy/nanoGPT

- requirements
  `pip3 install numpy torch transformers`

- run
  `python3 picogpt2.py --model gpt2 --start "Hi," --temperature 0.8 --topk 40`
"""
import argparse

import numpy as np
from transformers import GPT2Model, GPT2Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--start", default="\n", type=str)
parser.add_argument("--model", default="gpt2", type=str,
                    choices=("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"))
parser.add_argument("--max-new-tokens", default=100, type=int)
parser.add_argument("--temperature", default=1.0, type=float)
parser.add_argument("--topk", default=-1, type=int)
args = parser.parse_args()

# hparams
hparams = {
  "gpt2": dict(n_layer=12, n_head=12, n_embed=768),
  "gpt2-medium": dict(n_layer=24, n_head=16, n_embed=1024),
  "gpt2-large": dict(n_layer=36, n_head=20, n_embed=1280),
  "gpt2-xl": dict(n_layer=48, n_head=25, n_embed=1600)
}[args.model]
hparams.update(vocab_size=50257, context_len=1024)
# parameters
ws = {k: v.numpy() for k, v in GPT2Model.from_pretrained(args.model).state_dict().items()}

def layer_norm(x, w, b, eps=1e-5):
  mean = np.mean(x, axis=-1, keepdims=True)
  var = np.var(x, axis=-1, keepdims=True)
  return ((x - mean) / (var + eps)**0.5) * w + b

def linear(x, w, b):
  return x @ w + b

def gelu(x):
  return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def mlp(x, i):
  w, b = ws[f"h.{i}.mlp.c_fc.weight"], ws[f"h.{i}.mlp.c_fc.bias"]
  x = gelu(linear(x, w, b))
  w, b = ws[f"h.{i}.mlp.c_proj.weight"], ws[f"h.{i}.mlp.c_proj.bias"]
  return linear(x, w, b)

def softmax(x, axis=-1):
  x -= x.max(axis=axis, keepdims=True)
  x = np.exp(x, x)
  x /= x.sum(axis=axis, keepdims=True)
  return x

def mha(x, i):
  T, C = x.shape
  w, b = ws[f"h.{i}.attn.c_attn.weight"], ws[f"h.{i}.attn.c_attn.bias"]
  x = linear(x, w, b)

  n_head, hs = hparams["n_head"], C // hparams["n_head"]
  q, k, v = [np.transpose(h.reshape((T, n_head, hs)), (1,0,2)) for h in np.split(x, 3, axis=-1)]
  attn = softmax(q @ np.transpose(k, (0,2,1)) / hs**0.5 + (1 - np.tri(T)) * -1e20)
  x = np.transpose(attn @ v, (1,0,2)).reshape((T, C))

  w, b = ws[f"h.{i}.attn.c_proj.weight"], ws[f"h.{i}.attn.c_proj.bias"]
  x = linear(x, w, b)
  return x

def transformer_block(x, i):
  w, b = ws[f"h.{i}.ln_1.weight"], ws[f"h.{i}.ln_1.bias"]
  x = x + mha(layer_norm(x, w, b), i)
  w, b = ws[f"h.{i}.ln_2.weight"], ws[f"h.{i}.ln_2.bias"]
  x = x + mlp(layer_norm(x, w, b), i)
  return x

def gpt2(ids):
  wte, wpe = ws["wte.weight"], ws["wpe.weight"]
  x = wte[ids] + wpe[range(len(ids))]
  for i in range(hparams["n_layer"]):
    x = transformer_block(x, i)
  w, b = ws["ln_f.weight"], ws["ln_f.bias"]
  x = layer_norm(x, w, b)
  logits = x[-1, :] @ wte.T
  if args.temperature == 0:
    return np.argmax(logits)
  logits /= args.temperature
  if args.topk > 1:
    logits[np.argsort(logits)[:-args.topk]] = -float("inf")
  return np.random.choice(range(len(logits)), p=softmax(logits))

def sample():
  tokenizer = GPT2Tokenizer.from_pretrained(args.model)
  ids = tokenizer(args.start)["input_ids"]
  context_len = hparams["context_len"]
  for _ in range(args.max_new_tokens):
    ids_cond = ids if len(ids) <= context_len else ids[-context_len:]
    ids.append(gpt2(ids_cond))
    print("-"*30)
    print(tokenizer.decode(ids))


if __name__ == "__main__":
  sample()
