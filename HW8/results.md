# Homework 8 - Results

## Unformatted Output
```
=================================================================
GPU Configuration Information
=================================================================
GPU Device: NVIDIA GeForce RTX 4060 Ti
Compute Capability: 8.9
Number of SMs: 34
Max threads per SM: 1536
Max blocks per SM: 24
Thread warp size: 32

From this we can calculate an optimal launch configuration:
To maximize occupancy, we calculate BLOCK_SIZE as follows:
BLOCK_SIZE = max_threads_per_SM / max_blocks_per_SM = 64
GRID_SIZE = num_SMs * max_blocks_per_SM = 816 (persistent blocks)

Vector information:
Vector size: 40 MB
Number of float elements: 10485760
=================================================================

Running Task 1: CPU Reduction
CPU Result: 10485760.00 (expected: 10485760)
CPU-Avg-Time: 9.95159149ms

Running Task 2: GPU Atomic Cascaded Reduction (from HW7)
Reductions match.

GPU-Avg-Time (Atomic Cascaded): 0.32344794ms

Running Task 3: Harris Cascaded Algorithm with threadfence
Reductions match.

GPU-Avg-Time (Harris Cascaded + threadfence): 0.23119116ms
```

## Performance Summary (Formatted Output)

| Implementation                    | Average Time  | Speedup vs CPU |  
|-----------------------------------|---------------|----------------|  
| CPU Reduction (Sequential)        | 9.95 ms       | 1.0x           |  
| GPU Atomic Cascaded (HW7)         | 0.32 ms       | ~31.1x         |  
| GPU Harris Cascaded + threadfence | 0.23 ms       | ~43.3x         |  
