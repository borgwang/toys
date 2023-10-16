import os
import random
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
    subprocess.run(f"zstd -fq {inp} -o {out}", shell=True)
    original_size = os.path.getsize(inp)
    zst_size = os.path.getsize(out)

  # The theoretical maximum compression is the entropy of the data, i.e. H = -sum(p*log(p))
  theoretical_size = -sum(len(data)*p*np.log2(p) for p in probs)
  theoretical_size /= 8  # bits to bytes

  print(f"original_size: {original_size:,} B")
  print(f"zst_size: {zst_size:,} B")
  print(f"theoretical: {int(theoretical_size):,} B")

def compare_from_data(data):
  probs = [data.count(v) for v in vocab]
  probs = [p/sum(probs) for p in probs]
  compare(data, probs)

def compare_from_probs(probs):
  data = "".join(np.random.choice(vocab, N, p=probs))
  compare(data, probs)

# compression ratio is ~0.64 for random string of vocabulary with 36 characters
print("uniform distribution")
probs = [1] * len(vocab)
probs = [p/sum(probs) for p in probs]
compare_from_probs(probs)
print("----")

# can compress more efficiently if the character distribution is non-uniform
print("one-hot (nearly) distribution")
probs = [1] * len(vocab)
probs[0] *= 10000
probs = [p/sum(probs) for p in probs]
compare_from_probs(probs)

# we can surpass the shannon entropy if characters are not i.i.d.
print("---iid---")
raw_data = np.random.choice(vocab, N)
compare_from_data("".join(raw_data))
print("---non-iid---")
compare_from_data("".join(sorted(raw_data)))

