"""Reference: https://github.com/openai/tiktoken/blob/main/tiktoken/_educational.py"""

from collections import Counter

import regex


def preprocess(inputs:str, pattern:str) -> list[tuple[bytes, ...]]:
  words:list[str] = regex.findall(pattern, inputs)
  print(f"split to {len(words)} words. {len(set(words))} unique word.")
  return [tuple(c.to_bytes() for c in word.encode("utf-8")) for word in words]

def merge_pair(byte_arr:tuple[bytes], pair:tuple[bytes, bytes]) -> tuple[bytes, ...]:
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

def train(word_bytes:list[tuple[bytes, ...]], vocab_size:int, fast:bool=True) -> dict[bytes, int]:
  # the initial vocabulary contains 256 byte values
  vocab = {i.to_bytes(): i for i in range(2**8)}

  pair_cnt = Counter(t for ba in word_bytes for t in zip(ba[:-1], ba[1:]))
  while len(vocab) < vocab_size and pair_cnt:
    # most common pair
    pair = pair_cnt.most_common(1)[0][0]
    token_bytes = pair[0] + pair[1]
    # add to vocab
    vocab[token_bytes] = len(vocab)

    if fast: # modify pair_cnt inplace if fast=True
      for ba in word_bytes:
        for l in range(len(ba) - 1):
          if (ba[l], ba[l+1]) != pair:
            continue
          r = l + 2
          if ba[l-2:l] != pair:
            if l > 0:
              pair_cnt[(ba[l-1], pair[0])] -= 1
              pair_cnt[(ba[l-1], token_bytes)] += 1
          if ba[r:r+2] != pair:
            if r < len(ba):
              pair_cnt[(pair[1], ba[r])] -= 1
              pair_cnt[(token_bytes, ba[r])] += 1
          else:
            pair_cnt[(pair[1], pair[0])] -= 1
            pair_cnt[(token_bytes, token_bytes)] += 1
      # pop the most frequent pair in pair_cnt
      pair_cnt.pop(pair)

    # merge word bytes
    for i, ba in enumerate(word_bytes):
      word_bytes[i] = merge_pair(ba, pair)

    if not fast: # simply re-calculate pair_cnt if fast=False
      pair_cnt = Counter(t for ba in word_bytes for t in zip(ba[:-1], ba[1:]))

  return vocab

def encode(word_bytes:list[tuple[bytes, ...]], vocab:dict[bytes, int]) -> list[int]:
  tokens:list[int] = []
  for ba in word_bytes:
    while True:
      min_rank, pair = len(vocab), None
      for p in zip(ba[:-1], ba[1:]):
        token = vocab.get(p[0]+p[1])
        if token is not None and token < min_rank:
          min_rank, pair = token, p
      if min_rank == len(vocab):
        break
      ba = merge_pair(ba, pair)
    tokens.extend(vocab[b] for b in ba)
  return tokens

def decode(tokens:list[int], vocab:dict[bytes, int]) -> str:
  decoder = {i: b for b, i in vocab.items()}
  decoded_bytes =  b"".join(decoder[t] for t in tokens)
  # replace the invalid bytes with "�"
  return decoded_bytes.decode("utf-8", errors="replace")


if __name__ == "__main__":
  with open("./data/TinyStoriesV2-GPT4-valid.txt", "r") as f:
    inputs = f.read()[:1024*128]

  vocab_size = 600

  gpt2_pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
  word_bytes = preprocess(inputs, gpt2_pattern)

  # train
  vocab = train(word_bytes, vocab_size=vocab_size, fast=True)
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
