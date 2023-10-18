import os
import string
import subprocess
import tempfile

import numpy as np

N = 1_000_000
vocab = list(string.ascii_uppercase+string.digits)

def compare(data, probs):
  with tempfile.TemporaryDirectory() as tmpdir:
    inp = os.path.join(tmpdir, "random.txt")
    out = os.path.join(tmpdir, "random.zst")
    with open(inp, "w") as f:
      f.write(data)
    subprocess.run(f"zstd -fq {inp} -o {out}", shell=True, check=True)
    original_size = os.path.getsize(inp)
    zst_size = os.path.getsize(out)

  # The theoretical maximum compression is the entropy of the data, i.e. H = -sum(p*log(p))
  theoretical_size = -sum(len(data)*p*np.log2(p) for p in probs)
  theoretical_size /= 8  # bits to bytes

  print(f"original_size: {original_size:,} B")
  print(f"zst_size: {zst_size:,} B ({100*zst_size/original_size:.2f}%)")
  print(f"theoretical: {int(theoretical_size):,} B ({100*theoretical_size/original_size:.2f}%)")

def compare_from_data(data):
  probs = [data.count(v) for v in vocab]
  probs = [p/sum(probs) for p in probs]
  compare(data, probs)

def compare_from_probs(probs):
  data = "".join(np.random.choice(vocab, N, p=probs))
  compare(data, probs)


if __name__ == "__main__":
  # compression ratio is ~0.64 for random string from a 36 characters vocabulary
  print("--- uniform distribution random characters ---")
  probs = [1] * len(vocab)
  probs = [p/sum(probs) for p in probs]
  compare_from_probs(probs)

  # can compress more efficiently if the character distribution is non-uniform
  print("--- (nearly) one-hot distribution ---")
  probs = [1] * len(vocab)
  probs[0] *= 10000
  probs = [p/sum(probs) for p in probs]
  compare_from_probs(probs)

  # we can surpass the shannon entropy if the characters are not i.i.d.
  print("--- non-iid ---")
  raw_data = np.random.choice(vocab, N)
  compare_from_data("".join(sorted(raw_data)))
