import math

from numba import jit, vectorize, int32, float32
from numpy import log, log2
from PIL import Image, ImageDraw
import time
from numba import jit, prange, cuda
import numpy as np

CHANNELS = 3

RE_START, RE_END = -2, 1
IM_START, IM_END = -1, 1


# @jit(nopython=True, cache=True)
# def mandelbrot(c, max_iter):
#     z = 0
#     n = 0
#     while abs(z) <= 2 and n < max_iter:
#         z = z * z + c
#         n += 1
#
#     if n == max_iter:
#         return max_iter
#
#     return n + 1 - log(log2(abs(z)))


@cuda.jit
def calc_mandelbrot(img, max_iter, width, height):
    x = cuda.blockIdx.x
    y = cuda.threadIdx.x

    # Mapping pixel to C
    creal = RE_START + x / img.shape[0] * (width - RE_START)
    cim = IM_START + y / img.shape[1] * (height - IM_START)

    # Initialisation of C and Z
    c = complex(creal, cim)
    z = complex(0, 0)

    # z = 0
    n = 0
    while abs(z) <= 2 and n < max_iter:
        z = z * z + c
        n += 1

    if n == max_iter:
        m = max_iter
    else:
        m = n + 1 - math.log(math.log(abs(z)))/math.log(2)

    img[x][y][0], img[x][y][1], img[x][y][2] = int(255 * m / max_iter), 255, 255 if m < max_iter else 0


def draw_mandelbrot(draw, img, width, height):
    for x in range(0, width):
        for y in range(0, height):
            draw.point([x, y], (img[x][y][0], img[x][y][1], img[x][y][2]))


def draw_mandelbrot_set(max_iter, width, height, img_output_file):
    im = Image.new('HSV', (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    np.arange(0, 5, 0.5, dtype=int)

    img = np.zeros((width, height, CHANNELS), np.int32)

    start = time.time()
    calc_mandelbrot[width, height](img, max_iter, width, height)
    end = time.time()

    print(f'GPU version {max_iter}: Elapsed (with compilation) = {(end - start)}')

    draw_mandelbrot(draw, img, width, height)
    im.convert('RGB').save(img_output_file, 'PNG')
    return end - start
