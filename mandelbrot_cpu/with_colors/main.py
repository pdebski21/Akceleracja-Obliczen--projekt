from PIL import Image, ImageDraw
from mandelbrot import mandelbrot

MAX_ITER = 500

WIDTH, HEIGHT = 600, 400

RE_START, RE_END = -2, 1
IM_START, IM_END = -1, 1


def main():
    im = Image.new('HSV', (WIDTH, HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(im)

    for x in range(0, WIDTH):
        for y in range(0, HEIGHT):
            c = complex(RE_START + (x / WIDTH) * (RE_END - RE_START),
                        IM_START + (y / HEIGHT) * (IM_END - IM_START))
            m = mandelbrot(c, MAX_ITER)

            hue = int(255 * m / MAX_ITER)
            saturation = 255
            value = 255 if m < MAX_ITER else 0
            draw.point([x, y], (hue, saturation, value))

    im.convert('RGB').save('output.png', 'PNG')


if __name__ == "__main__":
    main()
