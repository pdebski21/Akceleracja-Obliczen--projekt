from PIL import Image, ImageDraw
from mandelbrot import mandelbrot
import time
from numba import jit, prange
import numpy as np

MAX_ITER = 500

WIDTH, HEIGHT, CHANNELS = 600, 400, 3

RE_START, RE_END = -2, 1
IM_START, IM_END = -1, 1


def create_hsv(m, max_iter):
    return int(255 * m / MAX_ITER), 255, 255 if m < MAX_ITER else 0


@jit(cache=True)
def calc_mandelbrot(img):
    for x in prange(0, WIDTH):
        for y in prange(0, HEIGHT):
            c = complex(RE_START + (x / WIDTH) * (RE_END - RE_START),
                        IM_START + (y / HEIGHT) * (IM_END - IM_START))
            m = mandelbrot(c, MAX_ITER)

            img[x][y][0], img[x][y][1], img[x][y][2] = int(255 * m / MAX_ITER), 255, 255 if m < MAX_ITER else 0

    return img


def draw_mandelbrot(draw, img):
    for x in range(0, WIDTH):
        for y in range(0, HEIGHT):
            draw.point([x, y], (img[x][y][0], img[x][y][1], img[x][y][2]))


def main():
    im = Image.new('HSV', (WIDTH, HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    np.arange(0, 5, 0.5, dtype=int)

    img = np.zeros((WIDTH, HEIGHT, CHANNELS), np.int32)

    start = time.time()
    calc_mandelbrot(img)
    end = time.time()

    print("Elapsed (with compilation) = %s" % (end - start))

    draw_mandelbrot(draw, img)
    im.convert('RGB').save('output.png', 'PNG')


if __name__ == "__main__":

    main()
