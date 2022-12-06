import csv


def write_to_csv(out_file_name, resolution,  time_cpu, time_cpu_parallel, time_gpu, time_gpu_withoutCompilation):
    out_file = open(out_file_name, 'w', newline='')
    headers = ['instance_size', 'resolution', 'time_cpu', 'time_cpu_parallel', 'time_gpu', 'time_gpu_only_computation']
    writer = csv.DictWriter(out_file, delimiter=';', lineterminator='\n', fieldnames=headers)

    writer.writeheader()

    for i in range(len(time_cpu)):
        writer.writerow({'instance_size': list(time_cpu.keys())[i],
                         'resolution': str(resolution),
                         'time_cpu': str(list(time_cpu.values())[i]).replace('.', ','),
                         'time_cpu_parallel': str(list(time_cpu_parallel.values())[i]).replace('.', ','),
                         'time_gpu': str(list(time_gpu.values())[i]).replace('.', ','),
                         'time_gpu_only_computation': str(list(time_gpu_withoutCompilation.values())[i]).replace('.', ',')
                         })

    out_file.close()
