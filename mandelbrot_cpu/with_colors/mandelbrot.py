from math import log, log2
from PIL import Image, ImageDraw
import time

RE_START, RE_END = -2, 1
IM_START, IM_END = -1, 1


def mandelbrot(c, max_iter):
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iter:
        z = z * z + c
        n += 1

    if n == max_iter:
        return max_iter

    return n + 1 - log(log2(abs(z)))


def draw_mandelbrot_set(MAX_ITER, WIDTH, HEIGHT, img_output_file):
    im = Image.new('HSV', (WIDTH, HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(im)

    start = time.time()
    for x in range(0, WIDTH):
        for y in range(0, HEIGHT):
            c = complex(RE_START + (x / WIDTH) * (RE_END - RE_START),
                        IM_START + (y / HEIGHT) * (IM_END - IM_START))
            m = mandelbrot(c, MAX_ITER)

            hue = int(255 * m / MAX_ITER)
            saturation = 255
            value = 255 if m < MAX_ITER else 0
            draw.point([x, y], (hue, saturation, value))
    end = time.time()

    print(f'CPU version {MAX_ITER}: Elapsed = {(end-start)}')
    im.convert('RGB').save(img_output_file, 'PNG')
    return end-start
