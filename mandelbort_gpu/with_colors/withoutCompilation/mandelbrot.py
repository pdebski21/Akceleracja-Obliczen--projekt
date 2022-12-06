import math

from PIL import Image, ImageDraw
import time
from numba import cuda
import numpy as np

CHANNELS = 3

RE_START, RE_END = -2, 1
IM_START, IM_END = -1, 1

@cuda.jit
def calc_mandelbrot(img, max_iter, width, height):
    x = cuda.blockIdx.x
    y = cuda.threadIdx.x

    creal = RE_START + x / width * (RE_END - RE_START)
    cim = IM_START + y / height * (IM_END - IM_START)

    c = complex(creal, cim)
    z = complex(0, 0)

    for n in range(max_iter):
        z = z * z + c
        # If unbounded: save iteration count and break
        if z.real * z.real + z.imag * z.imag > 4.0:
            # Smooth iteration count
            img[x][y][0], img[x][y][1], img[x][y][2] = int(255 * (n + 1 - math.log(math.log(abs(z)))/math.log(2)) / max_iter), 255, 255
            break

def draw_mandelbrot(draw, img, width, height):
    for x in range(0, width):
        for y in range(0, height):
            draw.point([x, y], (img[x][y][0], img[x][y][1], img[x][y][2]))


def draw_mandelbrot_set(max_iter, width, height, img_output_file):
    im = Image.new('HSV', (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    np.arange(0, 5, 0.5, dtype=int)

    img = np.zeros((width, height, CHANNELS), np.int32)


    calc_mandelbrot[width, height](img, max_iter, width, height)
    cuda.synchronize()
    start = time.time()
    calc_mandelbrot[width, height](img, max_iter, width, height)
    cuda.synchronize()
    end = time.time()

    print(f'GPU version {max_iter}: Elapsed (with compilation) = {(end - start)}')

    draw_mandelbrot(draw, img, width, height)
    im.convert('RGB').save(img_output_file, 'PNG')
    return end - start
