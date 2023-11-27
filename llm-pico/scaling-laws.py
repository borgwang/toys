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

  transformer = hp["n_layer"] * (ln + attn + mlp)

  ln_f = 2 * hp["n_embed"]
  # ignore parameters in the final softmax layer due to weight tying in gpt2
  return embedding + transformer + ln_f


def get_model_flops(hp):
  """
  for matrix multiplication A @ B = C, A: (B,M,K), B: (B,K,N), C: (B,M,N)
  the total FLOPs in a forward pass is B*(2*K*M*N) = 2BKMN
  in backward pass, we need to calculate 2 derivatives
    - dy/dA = dy/dC @ B.T, (B,M,N) @ (B,N,K), that is B*(2*N*M*K) FLOPs
    - dy/dB = A @ (dy/dC).T, (B,K,N) @ (B,N,M), that is also B*(2*N*M*K) FLOPs
  there are 4BKMN FLOPs in a backward pass
  """

  B = 1
  T = hp["context_len"]
  V = hp["vocab_size"]
  E = hp["n_embed"]

  # --- embedding ---
  # input -> (B, T)
  # wte: (V, E), wpe: (C, E)
  embedding = 2*T*V*E + 2*B*T*T*E + B*T*E

  # --- transformer block ----
  # input -> (B, T, E)
  # layernorm needs 8BTE
  #   - calculate mean and var -> 2BTE flops
  #   - (x-mean)/(var+eps)**0.5) * w + b -> 6BTE
  ln1 = 8*B*T*E
  mha = 0
  mha += 2*B*T*E*(3*E)  # c_attn
  mha += 2*B*T*E*T  # qk match, (B,T,E) @ (B,E,T) -> (B,T,T)
  mha += 2*B*T*T  # scale and mask
  mha += 5*B*T*T  # softmax (max/substract, exp, sum, divide)
  mha += 2*B*T*T*E # weighted sum v, (B,T,T) @ (B,T,E) -> (B,T,E)
  mha += 2*B*T*E*E  # c_proj (B,T,E) @ (E,E) -> (B,T,E)
  residual1 = B*T*E

  ln2 = 8*B*T*E
  mlp = 0
  mlp += 2*B*T*E*(4*E) # c_fc (B,T,E) @ (E,4E) -> (B,T,4E)
  mlp += 8*(B*T*4*E) # gelu
  mlp += 2*B*T*(4*E)*E # c_proj (B,T,4E) @ (4E,E)
  residual2 = B*T*E

  transformer = hp["n_layer"] * (ln1 + mha + residual1 + ln2 + mlp + residual2)

  ln_f = 8*B*T*E

  forward = embedding + transformer + ln_f
  backward = 2 * forward  # as in Kaplan et al. 2020
  return forward + backward


if __name__ == "__main__":
  print("--- parameters ---")
  for name, hparams in model_hparams.items():
    n_params = get_model_parameters(hparams)
    print(f"[{name}]#params: {n_params:,}")

    # verify our calculation with pytorch model checkpoint
    #from transformers import GPT2Model
    #model = GPT2Model.from_pretrained(name)
    #n_params_pt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #assert n_params == n_params_pt
    #print(f"[{name}]#params_from_pytorch: {n_params_pt:,}")


  print("--- FLOPs ---")
  for name, hparams in model_hparams.items():
    flops = get_model_flops(hparams)
    n_params = get_model_parameters(hparams)

    # for a modren transformer model,
    # most of parameters are weights used for matrix multiplication
    # and most of the FLOPs comes from matrix multiplication
    #
    # one can approximate the number of FLOPs by caculating FLOPs of
    # a single metirx multiplication, specifically (N,M) @ (M,K)
    # N is the total number of tokens, M is the embedding size, (M,K) is the shape of our large weight metrix
    # the approximate FLOPs is 6*N*M*K -> 6 * num_tokens * num_parameters
    approx_flops = 6 * n_params * hparams["context_len"]

    print(f"[{name}] flops: {flops/1e9:.2f} GFLOPs")
    print(f"[{name}] approx_flops: {approx_flops//1e9:.2f} GFLOPs")
