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


def draw_mandelbrot_set(max_iter, width, height, img_output_file):
    im = Image.new('HSV', (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(im)

    start = time.time()
    for x in range(0, width):
        for y in range(0, height):
            c = complex(RE_START + (x / width) * (RE_END - RE_START),
                        IM_START + (y / height) * (IM_END - IM_START))
            m = mandelbrot(c, max_iter)

            hue = int(255 * m / max_iter)
            saturation = 255
            value = 255 if m < max_iter else 0
            draw.point([x, y], (hue, saturation, value))
    end = time.time()

    print(f'CPU version {max_iter}: Elapsed = {(end-start)}')
    im.convert('RGB').save(img_output_file, 'PNG')
    return end-start
