import numpy as np

from array import ClArray
from ops import elemwise_op, reduce_op, matmul_op

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
