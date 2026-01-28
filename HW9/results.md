# Homework 9 - Results

## Unformatted output:
~~~
# Homework 9 - Results

## Unformatted output:
~~~
Running Task 1: CPU Scan
CPU-Avg-Time for 100000 elements: 0.22859573 ms
CPU-Avg-Time for 200000 elements: 0.45950413 ms
CPU-Avg-Time for 300000 elements: 0.70350170 ms
CPU-Avg-Time for 400000 elements: 0.95229149 ms
CPU-Avg-Time for 500000 elements: 1.20651722 ms
CPU-Avg-Time for 600000 elements: 1.50089264 ms
CPU-Avg-Time for 700000 elements: 1.74140930 ms
CPU-Avg-Time for 800000 elements: 1.96731091 ms
CPU-Avg-Time for 900000 elements: 2.31409073 ms
CPU-Avg-Time for 1000000 elements: 2.68909931 ms
Running Task 2: GPU Work-Efficient Scan
Reductions match. 
GPU-Avg-Time for 100000 elements: 0.24160695 ms
Reductions match. 
GPU-Avg-Time for 200000 elements: 0.17926383 ms
Reductions match. 
GPU-Avg-Time for 300000 elements: 0.18276095 ms
Reductions match. 
GPU-Avg-Time for 400000 elements: 0.17932415 ms
Reductions match. 
GPU-Avg-Time for 500000 elements: 0.18331409 ms
Reductions match. 
GPU-Avg-Time for 600000 elements: 0.62073302 ms
Reductions match. 
GPU-Avg-Time for 700000 elements: 0.64562798 ms
Reductions match. 
GPU-Avg-Time for 800000 elements: 0.63365221 ms
Reductions match. 
GPU-Avg-Time for 900000 elements: 0.62658620 ms
Reductions match. 
GPU-Avg-Time for 1000000 elements: 0.64174080 ms
~~~

## Formatted output
| num_elements | CPU (ms) | GPU (ms) | Speedup-Factor |
| ------------ | -------- | -------- | -------------- |
| 100000       | 0.2286   | 0.2416   | 0.95           |
| 200000       | 0.4595   | 0.1793   | 2.56           |
| 300000       | 0.7035   | 0.1828   | 3.85           |
| 400000       | 0.9523   | 0.1793   | 5.31           |
| 500000       | 1.2065   | 0.1833   | 6.58           |
| 600000       | 1.5009   | 0.6207   | 2.42           |
| 700000       | 1.7414   | 0.6456   | 2.70           |
| 800000       | 1.9673   | 0.6337   | 3.10           |
| 900000       | 2.3141   | 0.6266   | 3.69           |
| 1000000      | 2.6891   | 0.6417   | 4.19           |

## Plotted Execution times of GPU and CPU
![Execution times of the CPU and GPU scan algorithm for different number of elements. (Note: the y-axis are different for CPU and GPU)](./HW9/execution_times.png)  

## Plotted Speedup-Factor
![Speedup-Factor CPU/GPU of the scan algorithm for different number of elements](./HW9/speedup.png)  