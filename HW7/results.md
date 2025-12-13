#  Results
For this Ben Olschar, 108021211678 and Frederik Hüttemann, 108021215247 cooperated with eachother. We did implement the code on our own, but agreed to using one version. The discussion and results are made in cooperation.

## unformatted console output
~~~
Before starting, we need to get some information about the GPU (RTX 4060ti: 8.9 Compute Capability) launch configuration:
Number of SMs: 34
Max threads per SM: 1536
Max blocks per SM: 24
Thread warp size: 32
From this we can calculate an optimal launch configuration:
To maximize occupancy, we calculate BLOCK_SIZE as follows:
Suggested BLOCK_SIZE: 64
Suggested GRID_SIZE: 816
We also need to know the number of elements in our 40 MB float array: 
For 40 MB array size, use BLOCK_SIZE of 256 and GRID_SIZE of 544 for optimal performance.
A 40 MB array contains 10485760 float elements.


Running Task 1: CPU Reduction:
CPU Reduction Time: 21.819310 ms
Running Task 2: GPU Reduction Test with atomic add on global memory:
Results match.
GPU Reduction Time: 15.181743 ms
Running Task 3: GPU Reduction Test with cascaded reduction:
GPU Cascaded Reduction Time: 0.224458 ms
Results match.
~~~

## GPU Configuration:
| Parameter                    | Value       |
| ---------------------------- | ----------- |
| GPU                          | RTX 4060 Ti |
| Compute Capability           | 8.9         |
| Number of SMs                | 34          |
| Max threads per SM           | 1536        |
| Max blocks per SM            | 24          |
| Warp size                    | 32          |
| Suggested BLOCK_SIZE         | 64          |
| Suggested GRID_SIZE          | 816         |
| Array size                   | 40 MB       |
| Number of float elements     | 10,485,760  |


## Formatted results
| Task   | Description                              | Time (ms) | Status        |
| ------ | ---------------------------------------- | --------- | ------------- |
| Task 1 | CPU Reduction                            | 21.819310 | –             |
| Task 2 | GPU Reduction (atomicAdd, global memory) | 15.181743 | Results match |
| Task 3 | GPU Cascaded Reduction                   | 0.224458  | Results match |


