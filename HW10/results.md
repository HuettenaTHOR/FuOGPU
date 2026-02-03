# Homework 10 - Results

## Unformatted Output
~~~
Running Task 1: Single GPU Image convolution

H2D Data-Transfer: 1.05905533ms
GPU-Avg-Time (convolution) global memory: 2.03315711ms
D2H Data-Transfer: 0.80490112ms

Running Task 2: 100 sequential GPU Image convolution

H2D avg Data-Transfer: 1.05138302ms
H2D all Data-Transfer: 105.13830185ms

GPU-Avg-Time convolution sequential: 4.37034607ms
GPU-All-Time convolution sequential: 437.03460693ms

D2H avg Data-Transfer: 0.69632292ms
D2H all Data-Transfer: 69.63229179ms

Time for all (sequential): 611.80520058ms

Running Task 3: 100 concurrent GPU Image convolution

All-Time (Avg per Image): 4.10706997ms
All-Time (All Images): 410.70699692ms
Success!
~~~

## Task 1: Execution Time Measurements (Full HD: 1920 x 1080)

| Operation | Execution Time |
|-----------|----------------|
| Host-to-Device transfer | 1.06 ms |
| Kernel execution | 2.03 ms |
| Device-to-Host transfer | 0.80 ms |


## Performance Summary

| Implementation | Average Time | Speedup |
|----------------|--------------|---------|
| GPU - no streaming | 611.81 ms | baseline |
| GPU - with streaming | 410.71 ms | **1.49x** |  

