from transformers import GPT2Model

model_hparams = {
  "gpt2":        dict(n_layer=12, n_head=12, n_embed=768, vocab_size=50257, context_len=1024),
  #"gpt2-medium": dict(n_layer=24, n_head=16, n_embed=1024, vocab_size=50257, context_len=1024),
  #"gpt2-large":  dict(n_layer=36, n_head=20, n_embed=1280, vocab_size=50257, context_len=1024),
  #"gpt2-xl":     dict(n_layer=48, n_head=25, n_embed=1600, vocab_size=50257, context_len=1024)
}

def get_model_parameters(hp):
  # embedding
  wte = hp["vocab_size"] * hp["n_embed"]
  wpe = hp["context_len"] * hp["n_embed"]
  embedding = wte + wpe

  # transformer block
  ln = 2 * (2 * hp["n_embed"])  # two LayerNorm layer inside a block, each layer has weights and bias
  # attn
  c_attn = (hp["n_embed"] + 1) * 3 * hp["n_embed"]
  c_proj = (hp["n_embed"] + 1) * hp["n_embed"]
  attn = c_attn + c_proj
  # mlp
  c_fc = (hp["n_embed"] + 1) * 4 * hp["n_embed"]
  c_proj = (4 * hp["n_embed"] + 1) * hp["n_embed"]
  mlp = c_fc + c_proj
  block = ln + attn + mlp
  transformer = hp["n_layer"] * block

  ln_f = 2 * hp["n_embed"]
  # ignore parameters in the final softmax layer due to weight tying in gpt2
  return embedding + transformer + ln_f

for name, hparams in model_hparams.items():
  n_params = get_model_parameters(hparams)
  model = GPT2Model.from_pretrained(name)
  n_params_pt = sum(p.numel() for p in model.parameters() if p.requires_grad)
  assert n_params == n_params_pt
  # verify our calculation with pytorch model checkpoint
  print(f"#params_{name}: {n_params:,}")
  print(f"#params_{name}_from_pytorch: {n_params_pt:,}")


def get_model_flops(hp):
  # A @ B = C
  # A: (B,M,K), B: (B,K,N), C: (B,M,N)
  # the total FLOPs in a forward pass is B*(2*K*M*N) = 2BKMN
  # in backward pass, we have to calculate
  #   - dy/dA = dy/dC @ B.T, (B,M,N) @ (B,N,K), that is B*(2*N*M*K) FLOPs
  #   - dy/dB = A @ (dy/dC).T, (B,K,N) @ (B,N,M), that is B*(2*N*M*K) FLOPs
  # there are 4BKMN FLOPs in a backward pass
  pass
