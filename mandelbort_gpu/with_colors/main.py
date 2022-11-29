from PIL import Image, ImageDraw
import time
import numpy as np
from mandelbrot import calc_mandelbrot

MAX_ITER = 500

WIDTH, HEIGHT, CHANNELS = 600, 400, 3

RE_START, RE_END = -2, 1
IM_START, IM_END = -1, 1


def create_hsv(m, max_iter):
    return int(255 * m / MAX_ITER), 255, 255 if m < MAX_ITER else 0


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
    calc_mandelbrot[WIDTH, HEIGHT](img, MAX_ITER, WIDTH, HEIGHT)
    end = time.time()

    print("Elapsed (with compilation) = %s" % (end - start))

    draw_mandelbrot(draw, img)
    im.convert('RGB').save('output.png', 'PNG')


if __name__ == "__main__":
    main()
