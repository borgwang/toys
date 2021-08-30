import struct
import time

import numpy as np

"""
C code

float Q_rsqrt( float number )
{
    long i;
    float x2, y;
    const float threehalfs = 1.5F;

    x2 = number * 0.5F;
    y  = number;
    i  = * ( long * ) &y;                       // evil floating point bit level hacking
    i  = 0x5f3759df - ( i >> 1 );               // what the fuck?
    y  = * ( float * ) &i;
    y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration
    //    y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed
    return y;
}
"""


def intpr2long(n):
    return struct.unpack(">l", struct.pack(">f", n))[0]


def intpr2float(n):
    return struct.unpack(">f", struct.pack(">l", n))[0]


def normal_inv_sqrt(x):
    return 1. / x ** 0.5


def fast_inv_sqrt(x):
    y = intpr2float(0x5f3759df - (intpr2long(x) >> 1))
    return y * (1.5 - (0.5 * x * y * y))


def log2_approx(x):
    # interpret a float number y as an integer and do some scaling and shifting,
    # you will get good approximation of log2(y)
    return intpr2long(x) / 2 ** 23 + 0.043 - 127


def benchmark(control_fn, treatment_fn, size=100000):
    bias = []
    cost1, cost2 = 0, 0
    for _ in range(size):
        x = np.random.uniform(1, 100000)
        st = time.time()
        res1 = control_fn(x)
        et = time.time() - st
        cost1 += et

        st = time.time()
        res2 = treatment_fn(x)
        et = time.time() - st
        cost2 += et

        bias.append(np.abs((res2 - res1) / res1))

    print(f"control_fn cost: {cost1:.4f}")
    print(f"treatment_fn cost: {cost2:.4f}")
    print(f"bias cost: {np.mean(bias):.4f}")


benchmark(np.log2, log2_approx)
benchmark(normal_inv_sqrt, fast_inv_sqrt, size=1000000)
