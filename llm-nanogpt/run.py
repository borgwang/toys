import argparse
import dataclasses
import os
import importlib
import pickle
import time
from contextlib import nullcontext

import torch
import tiktoken

from dataset import Datasets
from lr import LRCosineAnnealing
from model import GPT
from utils import is_support_compile


def setup_torch(seed=31):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
  torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
  device_type = cfg_run.device_type
  if device_type == "cpu":
    return nullcontext()
  return torch.amp.autocast(device_type=device_type, dtype=getattr(torch, cfg_run.dtype))


def train():
  ckpt_dir = os.path.join(cfg_common.base_dir, cfg_run.dataset, "checkpoint")
  os.makedirs(ckpt_dir, exist_ok=True)
  ctx = setup_torch()
  dataset = Datasets[cfg_run.dataset](cfg)

  # update vocab_size from dataset
  cfg_model.vocab_size = dataset.meta["vocab_size"]

  # model
  iter_num = 0
  best_val_loss = float("inf")
  if cfg_run.init_from == "scratch":
    model = GPT(cfg_model)
  elif cfg_run.init_from == "resume":
    print(f"resume training from {ckpt_dir}")
    checkpoint = torch.load(os.path.join(ckpt_dir, "ckpt.pt"), map_location=cfg_run.device)
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["n_layer", "n_head", "n_embed", "context_len", "bias", "vocab_size"]:
      setattr(cfg_model, k, checkpoint["cfg"]["model"][k])
    model = GPT(cfg_model)
    model.load_state_dict(checkpoint["model"])
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
  elif cfg_run.init_from.startswith("gpt2"):
    model = GPT.from_pretrained(cfg_run.init_from, cfg_model)

  model.to(cfg_run.device)

  # optimizer
  optimizer = model.configure_optimizer(
      weight_decay=cfg_train.weight_decay,
      lr=cfg_train.lr,
      beta1=cfg_train.beta1,
      beta2=cfg_train.beta2
  )
  if cfg_run.init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])

  checkpoint = None

  if is_support_compile() and cfg_run.enable_compile:
    model = torch.compile(model)

  # lr schedule
  lr_scheduler = LRCosineAnnealing(
      lr=cfg_train.lr,
      min_lr=cfg_train.min_lr,
      warmup_iters=cfg_train.warmup_iters,
      decay_iters=cfg_train.lr_decay_iters,
  )

  @torch.no_grad()
  def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
      losses = torch.zeros(cfg_train.eval_iters)
      for k in range(cfg_train.eval_iters):
        X, Y = dataset.get_batch(split)
        with ctx:
          logits, loss = model(X, Y)
        losses[k] = loss.item()
      out[split] = losses.mean()
    model.train()
    return out

  # initialize a GradScaler. If enabled=False scaler is a no-op
  enable_grad_scaler = cfg_run.dtype == "float16" and cfg_run.device_type == "cuda"
  scaler = torch.cuda.amp.GradScaler(enabled=enable_grad_scaler)

  # main loop
  t0 = time.monotonic()
  X, Y = dataset.get_batch("train")

  while True:
    curr_lr = lr_scheduler(iter_num) if cfg_train.enable_lr_decay else cfg_train.lr
    for param_group in optimizer.param_groups:
      param_group["lr"] = curr_lr

    if iter_num % cfg_train.eval_interval == 0:
      losses = estimate_loss()
      print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

      if losses["val"] < best_val_loss or cfg_run.always_save_checkpoint:
        best_val_loss = losses["val"]
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter_num": iter_num,
            "best_val_loss": best_val_loss,
            "cfg": dataclasses.asdict(cfg),
        }
        print(f"saving checkpoint to {ckpt_dir}")
        torch.save(checkpoint, os.path.join(ckpt_dir, "ckpt.pt"))

    for micro_step in range(cfg_train.gradient_accumulation_steps):
      with ctx:
        logits, loss = model(X, Y)
        loss /= cfg_train.gradient_accumulation_steps
      # immediately async prefetch next batch while model is doing the forward pass
      X, Y = dataset.get_batch("train")
      scaler.scale(loss).backward()

    if (grad_clip := cfg_train.grad_clip) != 0:
      scaler.unscale_(optimizer)
      torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    t1 = time.monotonic()
    dt = t1 - t0
    t0 = t1
    if iter_num % cfg_run.log_interval == 0:
      lossf = loss.item() * cfg_train.gradient_accumulation_steps
      print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
    print(f"iter {iter_num}")

    iter_num += 1

    if iter_num > cfg_train.max_iters:
      break


def sample():
  ckpt_dir = os.path.join(cfg_common.base_dir, cfg_run.dataset, "checkpoint")
  ctx = setup_torch(cfg_sample.seed)
  # load model
  if cfg_run.init_from == "resume":
    checkpoint = torch.load(os.path.join(ckpt_dir, "ckpt.pt"), map_location=cfg_run.device)
    for k, v in checkpoint["cfg"]["model"].items():
      setattr(cfg_model, k, v)
    model = GPT(cfg_model)
    model.load_state_dict(checkpoint["model"])
  elif cfg_run.init_from.startswith("gpt2"):
    pass
  else:
    print(f"Invalid init_from ({cfg_run.init_from}) for sampling")
    exit(1)
  model.eval()
  model.to(cfg_run.device)
  if is_support_compile() and cfg_run.enable_compile:
    model = torch.compile(model)

  # encoder/decoder
  meta_path = os.path.join(cfg_common.base_dir, cfg_run.dataset, "data", "meta.pkl")
  if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
      meta = pickle.load(f)
      encode, decode = meta["encode"], meta["decode"]
  else:
    # assume gpt2 encodings by default
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda tokens: enc.encode(tokens, allowed_special={"<|endoftext|>"})
    decode = lambda ids: enc.decode(ids)

  # sample!
  start_ids = encode(cfg_sample.start)
  x = torch.tensor(start_ids, dtype=torch.long, device=cfg_run.device)[None, ...]
  with torch.no_grad():
    with ctx:
      for k in range(cfg_sample.num_samples):
        y = model.generate(x, cfg_sample.max_new_tokens, cfg_sample.temperature, cfg_sample.top_k)
        print(decode(y[0].tolist()))
        print("-------------")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("mode", type=str, choices=["train", "sample"])
  parser.add_argument("config", type=str,
      choices=["shakespeare_char", "shakespeare", "openwebtext"])
  meta_args, _ = parser.parse_known_args()
  cfg = importlib.import_module(f"conf.{meta_args.config}_conf").cfg
  # create argparse slots for arguments from config file
  sub_cfgs = [getattr(cfg, k) for k in dataclasses.asdict(cfg)]
  for sub_cfg in sub_cfgs:
    for field in dataclasses.fields(sub_cfg):
      parser.add_argument(f"--{field.name}", type=field.type)
  args = parser.parse_args()
  # parse and get arguments
  commandline_args = {k: v for k, v in vars(args).items() if v is not None}
  # overwrite default values in config file
  cfg.update(**commandline_args)

  cfg_run, cfg_train, cfg_sample, cfg_model, cfg_common = \
      cfg.run, cfg.training, cfg.sample, cfg.model, cfg.common

  # slightly different hparams for fine-tuning
  if cfg_run.init_from.startswith("gpt2"):
    cfg.setup_finetune()

  if meta_args.mode == "train":
    train()
  elif meta_args.mode == "sample":
    sample()
  else:
    pass
