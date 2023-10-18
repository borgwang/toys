import random
import string
import time
from collections import Counter
from decimal import Decimal, getcontext
from typing import List

getcontext().prec = 100000

random.seed(31)

def encode(inputs:List[str]) -> bytes:
  """encode inputs to bytes"""
  # calculate the target interval
  start, end = Decimal("0.0"), Decimal("1.0")
  for c in inputs:
    interval = end - start
    start = start + c_probs[c] * interval
    end = start + probs[c] * interval

  # find shortest binary representation for the interval
  bitarr = []
  while True:
    start, end = start+start, end+end
    if start <= 1 and end >= 1:
      bitarr.append("1")
      break
    flag = start > 1
    bitarr.append("1" if flag else "0")
    if flag:
      start, end = start-1, end-1

  # prepend an additional bit (1) to keep the leading zeros while converting to bytes
  bits = "".join(["1"] + bitarr)
  bytes_data = int(bits, 2).to_bytes((len(bits)+7)//8)
  # the header (4 bytes) implies data length
  header = len(inputs).to_bytes(4)
  return header + bytes_data

def decode(inputs:bytes) -> List[str]:
  """decode from bytes"""
  # parse data length and data bits
  length = int.from_bytes(inputs[:4])
  bits = bin(int.from_bytes(inputs[4:]))[2+1:]

  # recover a single decimal floating point number from binary bits
  v = Decimal("0.0")
  curr = Decimal("1.0")
  for b in bits:
    curr *= Decimal("0.5")
    if b == "1":
      v += curr

  def binary_search(w:Decimal, interval:Decimal, left:int, right:int) -> str:
    mid = int(left + (right - left) / 2)
    if mid == len(vocab) - 1:
      return vocab[mid]
    if w > c_probs[vocab[mid+1]] * interval:
      return binary_search(w, interval, mid+1, right)
    if w < c_probs[vocab[mid]] * interval:
      return binary_search(w, interval, left, mid-1)
    return vocab[mid]

  result = []
  start, end = Decimal("0.0"), Decimal("1.0")
  for _ in range(length):
    interval = end - start
    # find the symbol corresponding to the current value
    w = v - start
    c = binary_search(w, interval, 0, len(vocab) - 1)
    # update interval
    start = start + c_probs[c] * interval
    end = start + probs[c] * interval
    result.append(c)
  return result


if __name__ == "__main__":
  # create some random text
  N = 5000
  vocab = sorted(string.ascii_uppercase+string.digits)
  inputs = random.choices(vocab, k=N)

  # calculate probabilities and cumulative probabilities for each symbol
  counter = sorted(Counter(inputs).items())
  c_prob = Decimal("0.0")
  probs, c_probs = {}, {}
  for k, v in counter:
    prob = Decimal(v) / N
    probs[k] = prob
    c_probs[k] = c_prob
    c_prob += prob

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
