from numba import jit, vectorize, int32, float32
from numpy import log, log2
from PIL import Image, ImageDraw
import time
from numba import jit, prange
import numpy as np

CHANNELS = 3

RE_START, RE_END = -2, 1
IM_START, IM_END = -1, 1


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


@jit(cache=True)
def calc_mandelbrot(img, max_iter, width, height):
    for x in prange(0, width):
        for y in prange(0, height):
            c = complex(RE_START + (x / width) * (RE_END - RE_START),
                        IM_START + (y / height) * (IM_END - IM_START))
            m = mandelbrot(c, max_iter)

            img[x][y][0], img[x][y][1], img[x][y][2] = int(255 * m / max_iter), 255, 255 if m < max_iter else 0

    return img


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
    calc_mandelbrot(img, max_iter, width, height)
    end = time.time()

    print(f'GPU version {max_iter}: Elapsed (with compilation) = {(end - start)}')

    draw_mandelbrot(draw, img, width, height)
    im.convert('RGB').save(img_output_file, 'PNG')
    return end - start
