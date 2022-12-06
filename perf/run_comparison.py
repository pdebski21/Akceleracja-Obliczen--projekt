from configparser import ConfigParser
import writer
from mandelbrot_cpu.with_colors.mandelbrot import draw_mandelbrot_set as mandelbrot_cpu
from mandelbrot_cpu.with_colors.mandelbrot_parallel import draw_mandelbrot_set as mandelbrot_cpu_parallel
from mandelbort_gpu.with_colors.mandelbrot import draw_mandelbrot_set as mandelbrot_gpu
from mandelbort_gpu.with_colors.withoutCompilation.mandelbrot import draw_mandelbrot_set as mandelbrot_gpu_onlyComputation


def main():
    try:
        # Parameters from init file
        file = 'config.ini'
        config = ConfigParser()
        config.read(file)

        iterations = [int(x) for x in config['iterations']['iterations'].split(',')]
        print(f'Run for iterations: {iterations}')
        width = int(config['resolution']['width'])
        height = int(config['resolution']['height'])
        output_file = config['result']['output_calc']
        img_output_cpu = config['result']['img_output_cpu']
        img_output_cpu_parallel = config['result']['img_output_cpu_parallel']
        img_output_gpu = config['result']['img_output_gpu']
        img_output_gpu_withoutCompilation = config['result']['img_output_gpu_withoutCompilation']

        mandelbrot_cpu_results = {}
        mandelbrot_cpu_parallel_results = {}
        mandelbrot_gpu_results = {}
        mandelbrot_gpu_withoutCompilation_results = {}
        for iteration in iterations:
            cpu_time = mandelbrot_cpu(iteration, width, height, img_output_cpu)
            cpu_parallel_time = mandelbrot_cpu_parallel(iteration, width, height, img_output_cpu_parallel)
            gpu_time = mandelbrot_gpu(iteration, width, height, img_output_gpu)
            gpu_without_compilation_time = mandelbrot_gpu_onlyComputation(iteration, width, height,img_output_gpu_withoutCompilation)
            mandelbrot_cpu_results[iteration] = cpu_time
            mandelbrot_cpu_parallel_results[iteration] = cpu_parallel_time
            mandelbrot_gpu_results[iteration] = gpu_time
            mandelbrot_gpu_withoutCompilation_results[iteration] = gpu_without_compilation_time

        writer.write_to_csv(output_file, [width, height], mandelbrot_cpu_results,
                            mandelbrot_cpu_parallel_results, mandelbrot_gpu_results, mandelbrot_gpu_withoutCompilation_results)

    except OSError:
        print("Something wrong in init file")


if __name__ == "__main__":
    main()

