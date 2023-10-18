import os
import time

import numpy as np

os.environ['OMP_NUM_THREADS'] = '1'

DEBUG = 0

if DEBUG:
  N = 4
else:
  N = 512

def main():
  A = np.random.normal(0, 1, (N, N)).astype(np.float32)
  B = np.random.normal(0, 1, (N, N)).astype(np.float32)
  flop = 2 * N * N * N
  for _ in range(100):
    st = time.monotonic()
    #C = A @ B.T
    C = A @ B
    et = time.monotonic()
    s = et - st
    print(f"{flop/s * 1e-9:.2f} GFLOP/S, {s*1e3:.2f} ms")

  with open("/tmp/matmul", "wb") as f:
    f.write(A.data)
    f.write(B.data)
    f.write(C.data)


if __name__ == "__main__":
  main()
