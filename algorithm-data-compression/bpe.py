"""
Byte Pair Encoding (BPE) is a offline dictionary-based, grammar-based, multipass compression algorithm

BPE iteratively replaces the most frequent pair of bytes in a sequence with a single, unused byte.

see [Handbook of Data Compression](https://www.hlevkin.com/hlevkin/12videoprocDoc/Handbook%20of%20Data%20Compression,%205th%20Edition.pdf) Chapter 6 section 30
"""
import json
import re
import time
from collections import Counter


def encode_dict(d):
  return json.dumps(list(d.items())).encode()

def decode_dict(d):
  return dict(json.loads(d))

def encode(data:bytes, fast:bool=True) -> bytes:
  table = {i: (i, -2) for i in range(2**8)}
  for i in set(data):
    table[i] = (i, -1)

  bp_cnt = Counter(zip(data[:-1], data[1:]))
  while True:
    # find the most frequent byte pair
    bp, cnt = bp_cnt.most_common(1)[0]
    if cnt < 2:
      print("[INFO] no repeated byte pairs. exit loop")
      break
    # find an unused byte
    unused = None
    for i, (j, k) in table.items():
      if i == j and k == -2:
        unused = i
        break
    if unused is None:
      print("[INFO] no unused byte left. exit loop")
      break

    # replace the byte-pair to the unused byte
    table[unused] = bp
    replaced = data.replace(bytes(bp), bytes([unused]))

    # update bp_cnt
    if fast:
      # modify bp_cnt inplace
      for m in re.finditer(re.escape(bytes(bp)), data):
        l, r = m.span()
        # replace "ab" to "x" in "cab" -> decrease count("ca") and increse count("cx")
        if data[l-2:l] != bytes(bp):
          if l > 0:
            bp_cnt[(data[l-1], bp[0])] -= 1
            bp_cnt[(data[l-1], unused)] += 1
        # replace "ab" to "x" in "abc" -> decrease count("bc") and increse count("xc")
        if data[r:r+2] != bytes(bp):
          if r < len(data):
            bp_cnt[(bp[1], data[r])] -= 1
            bp_cnt[(unused, data[r])] += 1
        else:
          # replace "ab" to "x" in "abab" -> decrease count("ba") and increse count("xx")
          bp_cnt[(bp[1], bp[0])] -= 1
          bp_cnt[(unused, unused)] += 1
      # pop the target byte-pair in bp_cnt
      bp_cnt.pop(bp)
    else:
      # simply re-calculate bp_cnt
      bp_cnt = Counter(zip(replaced[:-1], replaced[1:]))

    data = replaced

  # encode table and data
  encoded_table = encode_dict(table)
  table_size = len(encoded_table).to_bytes(4)
  encoded = table_size + encoded_table + data
  return encoded

def decode(data:bytes) -> bytes:
  table_size = int.from_bytes(data[:4])
  table = decode_dict(data[4:4+table_size])
  data = data[4+table_size:]

  def parse(i):
    j, k = table[i]
    return i.to_bytes() if i==j and k==-1 else parse(j) + parse(k)

  return b"".join(parse(i) for i in data)


if __name__ == "__main__":
  # BPE compression works better for data tend to have many unused byte values, e.g. text data
  # we use the validation set of the [TinyStoriesV2](https://huggingface.co/datasets/roneneldan/TinyStories) dataset
  with open("./data/TinyStoriesV2-GPT4-valid.txt", "rb") as f:
    inputs = f.read()

  st = time.monotonic()
  encoded = encode(inputs)
  et = time.monotonic()
  print(f"encode time cost: {et-st:.4f}s")

  st = time.monotonic()
  decoded = decode(encoded)
  et = time.monotonic()
  print(f"decode time cost: {et-st:.4f}s")

  assert inputs == decoded

  print(f"input size: {len(inputs):,} B")
  print(f"compressed size: {len(encoded):,} B")
  print(f"compress factor: {100*len(encoded)/len(inputs):.2f}%")
