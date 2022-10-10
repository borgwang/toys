import copy

import numpy as np
import pyopencl

from utils import cl, prod

class ClArray:
    def __init__(self, data=None, shape=None, dtype=np.float32):
        self.shape, self.dtype = shape, dtype
        if isinstance(data, pyopencl.Buffer):
            self.buffer = data
            assert self.shape is not None, "Can not infer shape when initialize using clbuffer"
        else:
            if data is not None:
                data = np.asarray(data, dtype=self.dtype)
                self.shape = data.shape
            assert self.shape is not None, "Array shape is None!"
            self.buffer = cl.alloc_buffer(self.shape, self.dtype, data)

        # meta infos (https://numpy.org/doc/stable/dev/internals.html#numpy-internals)
        self.strides = tuple(prod(self.shape[i+1:]) for i in range(self.ndim))
        self._update_contiguity()
        self.offset = 0  # offset relative to the beginning of the buffer

    @property
    def ndim(self):
        return len(self.shape)

    def numpy(self):
        from ops import elemwise_op
        arr = elemwise_op("A", A=self)
        data = np.empty(arr.shape, dtype=arr.dtype)
        cl.enqueue("copy", data, arr.buffer, is_blocking=True)
        return data

    def _update_contiguity(self):
        # https://github.com/numpy/numpy/blob/4c60b3263ac50e5e72f6a909e156314fc3c9cba0/numpy/core/src/multiarray/flagsobject.c#L115
        self.c_contiguous = self.f_contiguous = True
        if not self.ndim: return
        nitems = 1
        for i in range(self.ndim-1, -1, -1):
            if self.shape[i] != 1:
                if self.strides[i] != nitems:
                    self.c_contiguous = False
                nitems *= self.shape[i]
        nitems = 1
        for i in range(self.ndim):
            if self.shape[i] != 1:
                if self.strides[i] != nitems:
                    self.f_contiguous = False
                nitems *= self.shape[i]

    def reshape(self, shape):
        if -1 in shape:
            size = prod(self.shape)
            assert shape.count(-1) <= 1, "Only one dimension can be inferred"
            axis = shape.index(-1)
            infer = prod([s for s in shape if s != -1])
            assert size % infer == 0, f"Shape {shape} invalid for size {size}"
            shape = (*shape[:axis], size // infer, *shape[axis+1:])
        shape = tuple(shape)
        assert prod(shape) == prod(self.shape), f"Can not reshape {self.shape} to {shape}"
        arr = elemwise_op(ElemwiseOps.NOOP, A=self) if not self.c_contiguous else self
        return self.view("reshape", shape, A=arr)

    def squeeze(self, axis=None):
        if axis is None:
            axis = [i for i, s in enumerate(self.shape) if s == 1]
        elif isinstance(axis, int):
            axis = [axis]
        axis = tuple([a if a != -1 else self.ndim - 1 for a in axis])
        shape = tuple([s for i, s in enumerate(self.shape) if i not in axis or self.shape[i] != 1])
        if shape == self.shape:
            return self
        return self.reshape(shape)

    def expand(self, shape):
        return self.view("expand", shape, A=self)

    def view(self, op, shape, **inp):
        x = list(inp.values())[0]
        inst = copy.copy(x)
        if op == "expand":
            strides = [0 if s1 < s2 else inst.strides[i] for i, (s1, s2) in enumerate(zip(inst.shape, shape))]
        elif op == "reshape":
            strides = (prod(shape[i+1:]) for i in range(len(shape))) if inst.c_contiguous else (prod(shape[:i]) for i in range(len(shape)))
        inst.shape, inst.strides = tuple(shape), tuple(strides)
        inst._update_contiguity()
        return inst

