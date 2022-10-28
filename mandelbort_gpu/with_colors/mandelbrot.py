from numba import jit, vectorize, int32, float32
from numpy import log, log2


@jit(nopython=True, cache=True)
def mandelbrot(c, max_iter):
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iter:
        z = z * z + c
        n += 1

    if n == max_iter:
        return max_iter

    return n + 1 - log(log2(abs(z)))

