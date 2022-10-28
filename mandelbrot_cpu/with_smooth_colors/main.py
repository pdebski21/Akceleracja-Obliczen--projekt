from PIL import Image, ImageDraw
from mandelbrot import mandelbrot
from collections import defaultdict
from math import floor, ceil

MAX_ITER = 500

WIDTH, HEIGHT = 600, 400

RE_START, RE_END = -2, 1
IM_START, IM_END = -1, 1


def linear_interpolation(color1, color2, t):
    return color1 * (1 - t) + color2 * t


def main():
    histogram = defaultdict(lambda: 0)
    values = {}
    for x in range(0, WIDTH):
        for y in range(0, HEIGHT):
            # Convert pixel coordinate to complex number
            c = complex(RE_START + (x / WIDTH) * (RE_END - RE_START),
                        IM_START + (y / HEIGHT) * (IM_END - IM_START))
            # Compute the number of iterations
            m = mandelbrot(c, 500)

            values[(x, y)] = m
            if m < MAX_ITER:
                histogram[floor(m)] += 1

    total = sum(histogram.values())
    hues = []
    h = 0
    for i in range(MAX_ITER):
        h += histogram[i] / total
        hues.append(h)
    hues.append(h)

    im = Image.new('HSV', (WIDTH, HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(im)

    for x in range(0, WIDTH):
        for y in range(0, HEIGHT):
            m = values[(x, y)]
            # The color depends on the number of iterations
            hue = 255 - int(255 * linear_interpolation(hues[floor(m)], hues[ceil(m)], m % 1))
            saturation = 255
            value = 255 if m < MAX_ITER else 0
            # Plot the point
            draw.point([x, y], (hue, saturation, value))

    im.convert('RGB').save('output.png', 'PNG')


if __name__ == "__main__":
    main()
