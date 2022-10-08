from functools import lru_cache
from functools import reduce

import numpy as np
import pyopencl
import pyopencl.clrandom as clrandom


def prod(data):
    return int(reduce(lambda a,b: a*b, data, 1))


class ClContext:
    def __init__(self):
        self.ctx, self.queue = None, None
        platform = pyopencl.get_platforms()[0]
        devices = platform.get_devices(device_type=pyopencl.device_type.GPU)
        if len(devices) == 0:
            devices = platform.get_devices(device_type=pyopencl.device_type.CPU)
        self.ctx = pyopencl.Context(devices)
        self.queue = pyopencl.CommandQueue(self.ctx)
        self.rng = clrandom.PhiloxGenerator(self.ctx, seed=0)

    @lru_cache(maxsize=None)
    def build(self, name, program):
        print(f"[DEBUG] program {name}: \n {program}")
        kernel = pyopencl.Program(self.ctx, program).build().__getattr__(name)
        return lambda *args: kernel(self.queue, *args)

    def alloc_local(self, size):
        return pyopencl.LocalMemory(size)

    def alloc_buffer(self, shape, dtype, hostbuf=None):
        size = int(dtype().itemsize * prod(shape))
        flags = pyopencl.mem_flags.READ_WRITE
        if hostbuf is not None:
            flags |= pyopencl.mem_flags.COPY_HOST_PTR
        return pyopencl.Buffer(self.ctx, flags, size, hostbuf=hostbuf)

    def enqueue(self, task, *args, **kwargs):
        getattr(pyopencl, f"enqueue_{task}")(self.queue, *args, **kwargs)

cl = ClContext()
