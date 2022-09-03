import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np

import time

N = 2048
A = np.random.randn(N, N).astype(np.float32)
B = np.random.randn(N, N).astype(np.float32)

flop = N*N*2*N
for _ in range(4):
    st = time.monotonic()
    C = A @ B
    et = time.monotonic()
    print(f"{(flop/1e9)/(et-st):3f} GFLOPs")

