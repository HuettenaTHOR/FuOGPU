#  Results
For this Ben Olschar, 108021211678 and Frederik Hüttemann, 108021215247 cooperated with eachother. We did implement the code on our own, but agreed to using one version. The discussion and results are made in cooperation.

## unformatted console output
~~~
Running Task 1: CPU Reduction
CPU-Avg-Time : 40.51799774ms

Running Task 2: GPU global memory Image upscale and convolution
Reductions match. 

GPU-Avg-Time global memory: 20.93264818ms
Running Task 3: GPU cascaded
Before starting, we need to get some information about the GPU launch configuration:
Number of SMs: 40
Max threads per SM: 1024
Max blocks per SM: 16
Thread warp size: 32
From this we can calculate an optimal launch configuration:
To maximize occupancy, we calculate BLOCK_SIZE as follows:
Suggested BLOCK_SIZE: 64
Suggested GRID_SIZE: 640
We also need to know the number of elements in our 40 MB float array: 
A 40 MB array contains 10485760 float elements.

Reductions match.
GPU-Avg-Time global memory: 0.19919896ms
~~~

## GPU Configuration:
| Parameter                    | Value                    |
| ---------------------------- | ------------------------ |
| GPU                          | NVIDIA GPU Tesla T4      |
| Compute Capability           | 7.5                      |
| Number of SMs                | 40                       |
| Max threads per SM           | 1024                     |
| Max blocks per SM            | 16                       |
| Warp size                    | 32                       |
| Suggested BLOCK_SIZE         | 64                       |
| Suggested GRID_SIZE          | 640                      |
| Array size                   | 40 MB                    |
| Number of float elements     | 10,485,760               |
| Optimized BLOCK_SIZE (40 MB) | 64                       |
| Optimized GRID_SIZE (40 MB)  | 640                      |



## Formatted results
| Task   | Description                           | Avg Time (ms) | Status           |
| ------ | ------------------------------------- | ------------- | ---------------- |
| Task 1 | CPU Reduction                         | 40.517998     | –                |
| Task 2 | GPU Reduction (global memory, atomic) | 20.932648     | Reductions match |
| Task 3 | GPU Cascaded Reduction                | 0.199199      | Reductions match |



