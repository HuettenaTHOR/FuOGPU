import re
from matplotlib import pyplot as plt

def parse_log_file(log_file):
    data = {
        'cpu_global_ilp1': [],
        'cpu_global_ilp4': [],
        'cpu_local_ilp1': [],
        'cpu_local_ilp4': [],
        'gpu_global_ilp1': [],
        'gpu_global_ilp4': [],
        'gpu_local_ilp1': [],
        'gpu_local_ilp4': []
    }
    x_axis = []
    # lines look like: "gpu/local/ilp1/16: 5370 MFLOP/s" (device/mem/ilp/size: value MFLOP/s)
    pattern = re.compile(r'^(cpu|gpu)/(global|local)/(ilp1|ilp4)/(\d+):\s*(\d+)\s*MFLOP/s$')
    
    with open(log_file, 'r') as f:
        for line in f:
            match = pattern.match(line.strip())
            if match:
                # device, memory type, ilp type, size, mflops
                device, mem_type, ilp_type, size, mflops = match.groups()
                key = f"{device}_{mem_type}_{ilp_type}"
                # keep only MFLOP value (log file is already ordered by size)
                data[key].append(int(mflops) / 1000)  # convert to GFLOP/s
                x_axis.append(int(size))
    x_axis = sorted(list(set(x_axis)))
    return data, x_axis


def plot_data(data, x_axis, cpu=False):
    # use size as x-axis    
    
    plt.figure(figsize=(12, 8))
    if cpu:
        plt.plot(x_axis, data['cpu_global_ilp1'], label='CPU Global ILP1')
        plt.plot(x_axis, data['cpu_global_ilp4'], label='CPU Global ILP4')
        plt.plot(x_axis, data['cpu_local_ilp1'], label='CPU Local ILP1')
        plt.plot(x_axis, data['cpu_local_ilp4'], label='CPU Local ILP4')
    else: 
        plt.plot(x_axis, data['gpu_global_ilp1'], label='GPU Global ILP1')
        plt.plot(x_axis, data['gpu_global_ilp4'], label='GPU Global ILP4')
        plt.plot(x_axis, data['gpu_local_ilp1'], label='GPU Local ILP1')
        plt.plot(x_axis, data['gpu_local_ilp4'], label='GPU Local ILP4')

    plt.xlabel('Threads')
    plt.ylabel('GFLOP/s')
    plt.title('Performance Comparison of Different ILPs and Memory Types')
    plt.legend()
    plt.grid(True)
    plt.savefig('HW3/performance_comparison.png')
    # save as pdf
    plt.savefig('HW3/performance_comparison.pdf')
    plt.show()

if __name__ == "__main__":
    log_file = "/mnt/e/dev/FuOGPU/HW3/results_final.txt"
    data, x_axis = parse_log_file(log_file)

    plot_data(data, x_axis, cpu=False)
