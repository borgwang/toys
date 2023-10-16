import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import is_pytorch2

class LayerNorm(nn.Module):
  def __init__(self, ndim, bias):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(ndim))
    self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

  def forward(self, x):
    return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.c_attn = nn.Linear(cfg.n_embed, 3*cfg.n_embed, bias=cfg.bias)
    self.c_proj = nn.Linear(cfg.n_embed, cfg.n_embed, bias=cfg.bias)
    self.attn_dropout = nn.Dropout(cfg.dropout)
    self.resid_dropout = nn.Dropout(cfg.dropout)
    self.n_head, self.n_embed, self.dropout = cfg.n_head, cfg.n_embed, cfg.dropout

    if not is_pytorch2():
      self.register_buffer("bias", torch.tril(torch.ones(cfg.context_len, cfg.context_len)).view(1, 1, cfg.context_len, cfg.context_len))

  def forward(self, x):
    B, T, C = x.size()
    q, k, v = self.c_attn(x).split(self.n_embed, dim=2)  # (B, T, C)
    hs = C // self.n_head
    # (B, nhead, T, hs)
    q = q.view(B, T, self.n_head, hs).transpose(1, 2)
    k = k.view(B, T, self.n_head, hs).transpose(1, 2)
    v = v.view(B, T, self.n_head, hs).transpose(1, 2)

    if is_pytorch2():
      y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0, is_causal=True)
    else:
      # (B, nhead, T, T)
      attn = (q @ k.transpose(2, 3)) / hs**0.5
      attn = attn.masked_fill(self.bias[:,:,:T,:T] == 0, float("-inf"))
      attn = F.softmax(attn, dim=-1)
      y = attn @ v
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    y = self.resid_dropout(self.c_proj(y))
    return y

class MLP(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.c_fc = nn.Linear(cfg.n_embed, 4*cfg.n_embed, bias=cfg.bias)
    self.gelu = nn.GELU()
    self.c_proj = nn.Linear(4*cfg.n_embed, cfg.n_embed, bias=cfg.bias)
    self.dropout = nn.Dropout(cfg.dropout)

  def forward(self, x):
    x = self.c_fc(x)
    x = self.gelu(x)
    x = self.c_proj(x)
    x = self.dropout(x)
    return x

class Block(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.ln_1 = LayerNorm(cfg.n_embed, bias=cfg.bias)
    self.attn = CausalSelfAttention(cfg)
    self.ln_2 = LayerNorm(cfg.n_embed, bias=cfg.bias)
    self.mlp = MLP(cfg)

  def forward(self, x):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x

class GPT(nn.Module):

  pretrain_model_config = {
      "gpt2": dict(n_layer=12, n_head=12, n_embed=768),
      "gpt2-medium": dict(n_layer=24, n_head=16, n_embed=1024),
      "gpt2-large": dict(n_layer=36, n_head=20, n_embed=1280),
      "gpt2-xl": dict(n_layer=48, n_head=25, n_embed=1600),
  }

  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    self.transformer = nn.ModuleDict(dict(
      wte=nn.Embedding(cfg.vocab_size, cfg.n_embed),
      wpe=nn.Embedding(cfg.context_len, cfg.n_embed),
      drop=nn.Dropout(cfg.dropout),
      h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
      ln_f=LayerNorm(cfg.n_embed, bias=cfg.bias)
    ))
    self.lm_head = nn.Linear(cfg.n_embed, cfg.vocab_size, bias=False)
    # weight tying (https://paperswithcode.com/method/weight-tying)
    self.transformer.wte.weight = self.lm_head.weight
    # init weights
    self.apply(self._init_weights)
    # apply special scaled init to the residual projections, per GPT-2 paper
    for pn, p in self.named_parameters():
      if pn.endswith("c_proj.weight"):
        torch.nn.init.normal_(p, mean=0.0, std=0.02/(2*cfg.n_layer)**0.5)

    print(f"number of parameters: {self.num_params/1e6:.2f}M")

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  @property
  def num_params(self):
    # subtract position embeddings (wpe). Token embeddings (wte) is counted due to the weight tying
    return sum(p.numel() for p in self.parameters()) - self.transformer.wpe.weight.numel()

  def crop_context_len(self, context_len):
    assert context_len <= self.cfg.context_len
    self.cfg.context_len = context_len
    self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:context_len])
    for block in self.transformer.h:
      if hasattr(block.attn, "bias"):
        block.attn.bias = block.attn.bias[:,:,:context_len,:context_len]

  def configure_optimizer(self, weight_decay, lr, beta1, beta2):
    param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
    decay_params = [p for p in param_dict.values() if p.dim() >= 2]
    nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
    optim_groups = [
      {"params": decay_params, "weight_decay": weight_decay},
      {"params": nodecay_params, "weight_decay": 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(beta1, beta2))
    return optimizer

  def forward(self, x, y=None):
    device = x.device
    b, t = x.size()
    assert t <= self.cfg.context_len, f"cannot forward sequence of length {t}, block size is only {self.cfg.context_len}"

    pos = torch.arange(0, t, dtype=torch.long, device=device)

    pos_emb = self.transformer.wpe(pos)
    tok_emb = self.transformer.wte(x)
    x = self.transformer.drop(tok_emb + pos_emb)
    for block in self.transformer.h:
      x = block(x)
    x = self.transformer.ln_f(x)

    if y is not None:
      logits = self.lm_head(x)  # B, T, V
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
    else:
      logits = self.lm_head(x[:, -1:, :])
      loss = None
    return logits, loss

  def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    context_len = self.cfg.context_len
    for _ in range(max_new_tokens):
      idx_cond = idx if idx.size(1) <= context_len else idx[:, -context_len:]
      logits, _ = self(idx_cond)
      logits = logits[:, -1, :] / temperature
      if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("inf")
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx

  @classmethod
  def from_pretrained(cls, model_type, cfg):
    assert model_type in cls.pretrain_model_config
    print(f"loading weights from pretrained gpt: {model_type}")
    for k, v in cls.pretrain_model_config[model_type].items():
      setattr(cfg, k, v)
    cfg.vocab_size = 50257
    cfg.context_len = 1024
    cfg.bias = True
    model = GPT(cfg)
    sd = model.state_dict()
    sd_keys = [k for k in sd.keys() if not k.endswith(".attn.bias")]

    from transformers import GPT2LMHeadModel
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = model_hf.state_dict()
    sd_hf_keys = [k for k in sd_hf.keys() if not k.endswith(".attn.masked_bias") and not k.endswith(".attn.bias")]
    assert len(sd_keys) == len(sd_hf_keys), f"mismatch keys {len(sd_keys)} != {len(sd_hf_keys)}"

    # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
    # this means that we have to transpose these weights when we import them
    transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]
    for k in sd_hf_keys:
      if any(k.endswith(w) for w in transposed):
        assert sd_hf[k].shape[::-1] == sd[k].shape
        with torch.no_grad():
          sd[k].copy_(sd_hf[k].t())
      else:
        assert sd_hf[k].shape == sd[k].shape
        with torch.no_grad():
          sd[k].copy_(sd_hf[k])
    return model

