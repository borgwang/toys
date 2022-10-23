import numpy as np
import time

from array import ClArray
from ops import elemwise_op, reduce_op, matmul_op

"""
A = ClArray(np.ones((2, 3)))
B = ClArray(np.ones((2, 3)))

# element-wise ops
res = elemwise_op("A+B", A=A, B=B)
print(res.numpy())
res = elemwise_op("A-B", A=A, B=B)
print(res.numpy())

res = elemwise_op("exp(A)", A=A)
print(res.numpy())

res = elemwise_op("A+B*exp(A)", A=A, B=B)
print(res.numpy())

# reduce op
res = reduce_op("sum", A=A)
print(res.numpy())

res = reduce_op("sum", axis=1, keepdims=True, A=A)
print(res.numpy())

# matmul op
np_A = np.arange(0, 6).reshape((2, 3))
np_B = np.arange(0, 6).reshape((3, 2))
print(np_A @ np_B)

A, B = ClArray(np_A), ClArray(np_B)
res = matmul_op(A=A, B=B)
print(res.numpy())
"""

np.random.seed(0)
shape = (10000, 10000)
np_A = np.random.uniform(0, 1, shape).astype(np.float32)
np_B = np.ones((1, 1)).astype(np.float32) * 9.999999747378752e-05
A = ClArray(np_A)
B = ClArray(np_B)
B = B.expand(shape)

st = time.monotonic()
res = elemwise_op("A*B", A=A, B=B)
print(time.monotonic() - st)
res = res.numpy()
print(res.sum())

# way faster
st = time.monotonic()
#res2 = elemwise_op("A*9.999999747378752e-05f", A=A)
res2 = elemwise_op("pow(A, 2.0f)", A=A)
print(time.monotonic() - st)
res2 = res2.numpy()
print(res2.sum())

