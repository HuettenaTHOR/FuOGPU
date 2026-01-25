# Homework 9 - Results

## Unformatted output:
~~~
Running Task 1: CPU Scan
CPU-Avg-Time for 100000 elements: 0.23069382 ms
CPU-Avg-Time for 200000 elements: 0.47559738 ms
CPU-Avg-Time for 300000 elements: 0.71887970 ms
CPU-Avg-Time for 400000 elements: 0.96769333 ms
CPU-Avg-Time for 500000 elements: 1.21459961 ms
CPU-Avg-Time for 600000 elements: 1.46558285 ms
CPU-Avg-Time for 700000 elements: 1.85110569 ms
CPU-Avg-Time for 800000 elements: 1.92718506 ms
CPU-Avg-Time for 900000 elements: 2.18701363 ms
CPU-Avg-Time for 1000000 elements: 2.43899822 ms
Running Task 2: GPU Work-Efficient Scan
GPU-Avg-Time for 100000 elements: 0.05141497 ms
GPU-Avg-Time for 200000 elements: 0.05809402 ms
GPU-Avg-Time for 300000 elements: 0.06259298 ms
GPU-Avg-Time for 400000 elements: 0.07170987 ms
GPU-Avg-Time for 500000 elements: 0.07586217 ms
GPU-Avg-Time for 600000 elements: 0.07606101 ms
GPU-Avg-Time for 700000 elements: 0.09311604 ms
GPU-Avg-Time for 800000 elements: 0.08684802 ms
GPU-Avg-Time for 900000 elements: 0.08895993 ms
GPU-Avg-Time for 1000000 elements: 0.09164810 ms
~~~

## Formatted output
| num_elements | CPU   | GPU   | Speedup-Factor |
|--------------|-------|-------|----------------|
| 100000       | 0.231 | 0.051 | 4.53           |
| 200000       | 0.476 | 0.058 | 8.21           |
| 300000       | 0.719 | 0.063 | 11.41          |
| 400000       | 0.968 | 0.072 | 13.44          |
| 500000       | 1.215 | 0.076 | 15.99          |
| 600000       | 1.466 | 0.076 | 19.29          |
| 700000       | 1.851 | 0.093 | 19.91          |
| 800000       | 1.927 | 0.087 | 22.15          |
| 900000       | 2.187 | 0.089 | 24.57          |
| 1000000      | 2.439 | 0.092 | 26.51          |

## Plotted Execution times of GPU and CPU
![Execution times of the CPU and GPU scan algorithm for different number of elements. (Note: the y-axis are different for CPU and GPU)](./HW9/execution_times.png)  

## Plotted Speedup-Factor
![Speedup-Factor CPU/GPU of the scan algorithm for different number of elements](./HW9/plt.png)  