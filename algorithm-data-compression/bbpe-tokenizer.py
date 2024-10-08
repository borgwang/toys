"""Reference: https://github.com/openai/tiktoken/blob/main/tiktoken/_educational.py"""

from collections import Counter

import regex


class BBPETokenizer:
  def __init__(self, pat_str:str):
    self.pat_str = pat_str
    self.vocab = None

  def _preprocess(self, inputs:str) -> list[tuple[bytes, ...]]:
    words:list[str] = regex.findall(self.pat_str, inputs)
    print(f"split to {len(words)} words. {len(set(words))} unique word.")
    return [tuple(c.to_bytes() for c in word.encode("utf-8")) for word in words]

  @staticmethod
  def _merge_pair(byte_arr:tuple[bytes, ...], pair:tuple[bytes, bytes]) -> tuple[bytes, ...]:
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

  def train(self, inputs:str, vocab_size:int):
    word_bytes = self._preprocess(inputs)
    # the initial vocabulary contains 256 byte values
    self.vocab = {i.to_bytes(): i for i in range(2**8)}

    pair_cnt = Counter(t for ba in word_bytes for t in zip(ba[:-1], ba[1:]))
    while len(self.vocab) < vocab_size and pair_cnt:
      # most common pair
      pair = pair_cnt.most_common(1)[0][0]
      token_bytes = pair[0] + pair[1]
      # add to vocab
      self.vocab[token_bytes] = len(self.vocab)

      # modify pair_cnt inplace
      pair_cnt.pop(pair)
      for ba in word_bytes:
        for l in range(len(ba) - 1):
          if (ba[l], ba[l+1]) != pair:
            continue
          r = l + 2
          if ba[l-2:l] != pair:
            # replace "ab" to "x" in "cab" -> decrease count("ca") and increse count("cx")
            if l > 0:
              pair_cnt[(ba[l-1], pair[0])] -= 1
              pair_cnt[(ba[l-1], token_bytes)] += 1
          if ba[r:r+2] != pair:
            # replace "ab" to "x" in "abc" -> decrease count("bc") and increse count("xc")
            if r < len(ba):
              pair_cnt[(pair[1], ba[r])] -= 1
              pair_cnt[(token_bytes, ba[r])] += 1
          else:
            # replace "ab" to "x" in "abab" -> decrease count("ba") and increse count("xx")
            pair_cnt[(pair[1], pair[0])] -= 1
            pair_cnt[(token_bytes, token_bytes)] += 1

      # merge word bytes
      for i, ba in enumerate(word_bytes):
        word_bytes[i] = self._merge_pair(ba, pair)

  def encode(self, inputs:str) -> list[int]:
    word_bytes = self._preprocess(inputs)
    tokens:list[int] = []
    for ba in word_bytes:
      while True:
        min_rank, pair = len(self.vocab), None
        for p in zip(ba[:-1], ba[1:]):
          token = self.vocab.get(p[0]+p[1])
          if token is not None and token < min_rank:
            min_rank, pair = token, p
        if min_rank == len(self.vocab):
          break
        ba = self._merge_pair(ba, pair)
      tokens.extend(self.vocab[b] for b in ba)
    return tokens

  def decode(self, tokens:list[int]) -> str:
    decoder = {i: b for b, i in self.vocab.items()}
    decoded_bytes =  b"".join(decoder[t] for t in tokens)
    # replace the invalid bytes with "�"
    return decoded_bytes.decode("utf-8", errors="replace")


if __name__ == "__main__":
  with open("./data/TinyStoriesV2-GPT4-valid.txt", "r") as f:
    inputs = f.read()[:1024*128]

  gpt2_pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
  tokenizer = BBPETokenizer(gpt2_pattern)
  tokenizer.train(inputs, vocab_size=600)
  print(f"vocab={tokenizer.vocab}")

  # encode
  inputs = "thousand of cities from home, wonder into the unknown"
  print(f"encoding_text='{inputs}'")
  tokens = tokenizer.encode(inputs)
  print(f"tokens={tokens}")

  # decode
  decoded = tokenizer.decode(tokens)
  assert inputs == decoded
