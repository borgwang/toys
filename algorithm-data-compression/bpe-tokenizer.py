"""
Reference: https://github.com/openai/tiktoken/blob/main/tiktoken/_educational.py

- 为什么 BPE 压缩会收到 unused byte 的约束，但是 BPE tokenize 的时候不会？

"""

from collections import Counter
from functools import lru_cache

import regex


def preprocess(inputs:str, pattern:str) -> list[list[bytes]]:
  words:list[str] = regex.findall(pattern, inputs)
  print(f"split to {len(words)} words. {len(set(words))} unique word.")
  word_bytes = [tuple(bytes([c]) for c in word.encode("utf-8")) for word in words]
  return word_bytes

@lru_cache(maxsize=None)
def merge_pair(byte_arr:tuple[bytes], pair:tuple[bytes, bytes]) -> tuple[bytes]:
  j = 0
  merged = []
  while j < len(byte_arr) - 1:
    if (byte_arr[j], byte_arr[j+1]) == pair:
      merged.append(pair[0] + pair[1])
      j += 2
    else:
      merged.append(byte_arr[j])
      j += 1
  if j == len(byte_arr) - 1:
    merged.append(byte_arr[j])
  return tuple(merged)

def train(word_bytes:list[tuple[bytes]], vocab_size:int) -> dict[bytes, int]:
  # train tokenizer, i.e. build a vocabulary from input data
  vocab = {i.to_bytes(): i for i in range(2**8)}
  while len(vocab) < vocab_size:
    counter = Counter(t for bs in word_bytes for t in zip(bs[:-1], bs[1:]))
    if not counter:
      print(f"No pair can be futher construct from data. current vocab_size={len(vocab)}")
      break
    pair, _ = counter.most_common(1)[0]
    token_bytes = pair[0] + pair[1]
    # add to vocab
    vocab[token_bytes] = len(vocab)
    # merge word bytes
    for i, bs in enumerate(word_bytes):
      word_bytes[i] = merge_pair(bs, pair)
  return vocab

def encode(word_bytes:list[tuple[bytes]], vocab:dict[bytes, int]) -> list[int]:
  tokens = []
  for bs in word_bytes:
    while True:
      min_rank, pair = len(vocab), None
      for p in zip(bs[:-1], bs[1:]):
        token = vocab.get(p[0]+p[1])
        if token is not None and token < min_rank:
          min_rank, pair = token, p
      if min_rank == len(vocab):
        break
      bs = merge_pair(bs, pair)
    tokens.extend(vocab[b] for b in bs)
  return tokens

def decode(tokens:list[int], vocab:dict[bytes, int]):
  decoder = {i: b for b, i in vocab.items()}
  decoded_bytes =  b"".join(decoder[t] for t in tokens)
  # replace the invalid bytes with "�"
  decoded = decoded_bytes.decode("utf-8", errors="replace")
  return decoded


if __name__ == "__main__":
  with open("./data/TinyStoriesV2-GPT4-valid.txt", "r") as f:
    inputs = f.read()[:1024*128]

  vocab_size = 600

  gpt2_pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
  word_bytes = preprocess(inputs, gpt2_pattern)

  # train
  vocab = train(word_bytes, vocab_size=vocab_size)
  print(f"vocab={vocab}")

  # encode
  inputs = "thousand of cities from home, wonder into the unknown"
  print(f"encoding_text='{inputs}'")
  word_bytes = preprocess(inputs, gpt2_pattern)
  tokens = encode(word_bytes, vocab)
  print(f"tokens={tokens}")

  # decode
  decoded = decode(tokens, vocab)
  assert inputs == decoded
