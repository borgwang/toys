import os
import pickle
import requests

import numpy as np
import tiktoken
import torch
from tqdm import tqdm

gpt2enc = tiktoken.get_encoding("gpt2")

class Dataset:
  def __init__(self, cfg):
    self.cfg = cfg

    # gpt2 encoding by default
    self.vocab_size = gpt2enc.n_vocab

    self.data_dir = os.path.join(cfg.common.base_dir, cfg.run.dataset, "data")
    os.makedirs(self.data_dir, exist_ok=True)

    self.train_path = os.path.join(self.data_dir, "train.bin")
    self.val_path = os.path.join(self.data_dir, "val.bin")
    if not (os.path.exists(self.train_path) and os.path.exists(self.val_path)):
      self.prepare()

    self.train_data = np.memmap(self.train_path, dtype=np.uint16, mode="r")
    self.val_data = np.memmap(self.val_path, dtype=np.uint16, mode="r")

  def encode(self, tokens, **kwargs):
    return gpt2enc.encode(tokens, **kwargs)

  def decode(self, ids, **kwargs):
    return gpt2enc.decode(ids, **kwargs)

  def get_batch(self, split):
    context_len = self.cfg.model.context_len
    data = self.train_data if split == "train" else self.val_data
    ix = torch.randint(len(data) - context_len, (self.cfg.training.batch_size,))
    x = torch.stack([torch.from_numpy((data[i: i+context_len]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1: i+1+context_len]).astype(np.int64)) for i in ix])
    device = self.cfg.run.device
    if "cuda" in device:
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

  @property
  def meta(self):
    with open(os.path.join(self.data_dir, "meta.pkl"), "rb") as f:
      meta = pickle.load(f)
    return meta


class ShakespeareDataset(Dataset):
  url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

  def _parse_if_needed(self, data):
    pass

  def prepare(self):
    # download the tiny shakespeare dataset
    input_file_path = os.path.join(self.data_dir, "input.txt")

    if not os.path.exists(input_file_path):
      with open(input_file_path, "w") as f:
        f.write(requests.get(self.url).text)

    with open(input_file_path, "r") as f:
      data = f.read()

    self._parse_if_needed(data)

    split = int(len(data) * 0.9)
    train_ids = self.encode(data[:split], disallowed_special=())
    val_ids = self.encode(data[split:], disallowed_special=())
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    np.array(train_ids, dtype=np.uint16).tofile(self.train_path)
    np.array(val_ids, dtype=np.uint16).tofile(self.val_path)

    # save the meta information as well, to help us encode/decode later
    meta = {"vocab_size": self.vocab_size, "encode": self.encode, "decode": self.decode}
    with open(os.path.join(self.data_dir, "meta.pkl"), "wb") as f:
      pickle.dump(meta, f)


class ShakespeareCharDataset(ShakespeareDataset):

  def _parse_if_needed(self, data):
    chars = sorted(list(set(data)))
    self.vocab_size = len(chars)
    # create a mapping from characters to integers
    self.stoi = {ch: i for i, ch in enumerate(chars)}
    self.itos = {i: ch for i, ch in enumerate(chars)}

  def encode(self, tokens, **kwargs):
    return [self.stoi[t] for t in tokens]

  def decode(self, ids, **kwargs):
    return "".join([self.itos[i] for i in ids])


class OpenwebtextDataset(Dataset):
  """
  # train.bin is ~17GB, val.bin ~8.5MB
  # train has ~9B tokens (9,035,582,198)
  # val has ~4M tokens (4,434,897)
  """
  num_proc = 8

  def tokenize_example(self, example):
    ids = self.encode(example["text"], disallowed_special=())
    ids.append(gpt2enc.eot_token)
    return {"ids": ids, "len": len(ids)}

  def prepare(self):
    from datasets import load_dataset
    dataset = load_dataset("openwebtext", num_proc=self.num_proc,
                           cache_dir="/data/workspace/.cache/huggingface/datasets")

    # owt by default only contains the 'train' split, so create a test split
    dataset = dataset["train"].train_test_split(test_size=0.0005, seed=31, shuffle=True)
    dataset["val"] = dataset.pop("test")
    dataset = dataset.map(
        self.tokenize_example,
        remove_columns=["text"],
        num_proc=self.num_proc,
    )

    total_batches = 1024
    for split, dset in dataset.items():
      path = self.train_path if split == "train" else self.val_path
      arr_len = np.sum(dset["len"], dtype=np.uint64)
      arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(arr_len,))
      idx = 0
      for batch_idx in tqdm(range(total_batches), desc=f"writing {path}"):
        # Batch together samples for faster write
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format("numpy")
        arr_batch = np.concatenate(batch["ids"])
        ## Write into mmap
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
      arr.flush()

    # save the meta information as well, to help us encode/decode later
    meta = {"vocab_size": self.vocab_size, "encode": self.encode, "decode": self.decode}
    with open(os.path.join(self.data_dir, "meta.pkl"), "wb") as f:
      pickle.dump(meta, f)



Datasets = {
    "shakespeare-char": ShakespeareCharDataset,
    "shakespeare": ShakespeareDataset,
    "openwebtext": OpenwebtextDataset
}

