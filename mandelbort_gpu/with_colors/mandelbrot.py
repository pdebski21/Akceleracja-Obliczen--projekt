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
def calc_mandelbrot(img, MAX_ITER, WIDTH, HEIGHT):
    for x in prange(0, WIDTH):
        for y in prange(0, HEIGHT):
            c = complex(RE_START + (x / WIDTH) * (RE_END - RE_START),
                        IM_START + (y / HEIGHT) * (IM_END - IM_START))
            m = mandelbrot(c, MAX_ITER)

            img[x][y][0], img[x][y][1], img[x][y][2] = int(255 * m / MAX_ITER), 255, 255 if m < MAX_ITER else 0

    return img


def draw_mandelbrot(draw, img, WIDTH, HEIGHT):
    for x in range(0, WIDTH):
        for y in range(0, HEIGHT):
            draw.point([x, y], (img[x][y][0], img[x][y][1], img[x][y][2]))


def draw_mandelbrot_set(MAX_ITER, WIDTH, HEIGHT, img_output_file):
    im = Image.new('HSV', (WIDTH, HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    np.arange(0, 5, 0.5, dtype=int)

    img = np.zeros((WIDTH, HEIGHT, CHANNELS), np.int32)

    start = time.time()
    calc_mandelbrot(img, MAX_ITER, WIDTH, HEIGHT)
    end = time.time()

    print(f'GPU version {MAX_ITER}: Elapsed (with compilation) = {(end - start)}')

    draw_mandelbrot(draw, img, WIDTH, HEIGHT)
    im.convert('RGB').save(img_output_file, 'PNG')
    return end - start
