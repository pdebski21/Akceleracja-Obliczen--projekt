from PIL import Image, ImageDraw
from mandelbrot import mandelbrot

MAX_ITER = 500

WIDTH, HEIGHT = 600, 400

RE_START, RE_END = -2, 1
IM_START, IM_END = -1, 1


def main():
    im = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(im)

    for x in range(0, WIDTH):
        for y in range(0, HEIGHT):
            c = complex(
                RE_START + (x / WIDTH) * (RE_END - RE_START),
                IM_START + (y / HEIGHT) * (IM_END - IM_START),
            )
            m = mandelbrot(c, MAX_ITER)

            color = 255 - int(m * 255 / MAX_ITER)
            draw.point([x, y], (color, color, color))

    im.save("output.png", "PNG")


if __name__ == "__main__":
    main()
