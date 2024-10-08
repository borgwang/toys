import os
import time

import numpy as np

threads = 1
os.environ["OMP_NUM_THREADS"] = str(threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
os.environ["MKL_NUM_THREADS"] = str(threads)

DEBUG = 0

if DEBUG:
  N = 4
else:
  N = 512

def main():
  A = np.random.normal(0, 1, (N, N)).astype(np.float32)
  B = np.random.normal(0, 1, (N, N)).astype(np.float32)
  repeat = 1000
  flop = 2 * N * N * N * repeat
  cost = 0  # ns
  for _ in range(repeat):
    st = time.perf_counter_ns()
    #C = A @ B.T
    C = A @ B
    et = time.perf_counter_ns()
    cost += (et-st)
  print(f"{flop/cost:.2f} GFLOP/S, {cost/1e6:.2f} ms")

  with open("/tmp/matmul", "wb") as f:
    f.write(A.data)
    f.write(B.data)
    f.write(C.data)


if __name__ == "__main__":
  main()
