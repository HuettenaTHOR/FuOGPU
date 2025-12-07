#  Results
For this Ben Olschar, 108021211678 and Frederik Hüttemann, 108021215247 cooperated with eachother. We did implement the code on our own, but agreed to using one version. The discussion and results are made in cooperation.

## unformatted console output
~~~
Running Task 1: CPU Upscaling + Convolution
CPU Upscaling + Convolution Time: 13.327447 seconds
Running Task 2: GPU Upscaling + Convolution
GPU Upscaling + Convolution (global memory) Time: 5.629340 ms 
Results match.
Running Task 3: GPU Upscaling + Convolution with Constant Memory
GPU Upscaling + Convolution (constant memory) Time: 3.967050 ms 
Results match.
Running Task 4: GPU Upscaling with Texture Memory + Convolution with Constant Memory
GPU Upscaling + Convolution (constant memory) Time: 3.856604 ms 
Results match.
~~~

## Formatted results

| Task | Implementation | Total Time | Speedup vs CPU |
|------|---------------|------------|----------------|
| 1 | CPU Upscaling + Convolution | 13327.45 ms | 1.00x (baseline) |
| 2 | GPU (Global Memory) | 5.63 ms | 2367.37x |
| 3 | GPU (Constant Memory) | 3.97 ms | 3358.70x |
| 4 | GPU (Texture + Constant Memory) | 3.86 ms | 3453.77x |

