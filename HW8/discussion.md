# Homework 8 - Discussion

## What We Expected

1. **CPU Reduction Performance**: We expected the CPU reduction to be relatively slow since it's a sequential operation that cannot utilize the parallel nature of the GPU. The CPU must iterate through all 10+ million elements one by one.

2. **GPU Atomic Cascaded (HW7)**: We expected decent speedup over CPU, but with some overhead from atomic operations. The atomic operations, while providing correctness, serialize access to shared and global memory locations which limits parallelism.

3. **Harris Cascaded Algorithm with threadfence**: We expected this implementation to be faster than the atomic approach because:
   - **Sequential addressing** eliminates bank conflicts in shared memory
   - **Unrolling the last warp** with volatile keyword removes unnecessary `__syncthreads()` calls within a warp
   - **Cascaded algorithm** (accumulating in registers via grid-stride loops) reduces shared memory traffic
   - **Persistent blocks** (GRID_SIZE = #SMs × #blocks_per_SM) ensures all SMs are fully occupied
   - **`__threadfence()`** avoids launching a second kernel for final reduction, eliminating kernel launch overhead

## What We Observed

1. **CPU Reduction**: ~10 ms for 40MB (10,485,760 floats) - sequential performance as expected.

2. **GPU Atomic Cascaded**: ~0.32 ms providing ~31x speedup over CPU. This is already a significant improvement, utilizing thread coarsening and atomic operations.

3. **Harris Cascaded with threadfence**: ~0.23 ms providing ~43x speedup over CPU and ~1.4x faster than atomic cascaded.

## Analysis

**Does the observation match the expectation?**

Yes, the Harris cascaded algorithm with `__threadfence()` outperforms the atomic cascaded approach as expected.

**Why is Harris Cascaded faster than Atomic Cascaded?**

1. **Reduced atomic contention**: The Harris algorithm uses tree-based reduction within each block (with `warpReduce` using volatile shared memory), which is more efficient than every thread calling `atomicAdd` to a single shared variable.

2. **Warp-level efficiency**: The `warpReduce` function with volatile keyword exploits SIMD execution within a warp - no synchronization needed since all threads in a warp execute in lockstep.

3. **Single-kernel final reduction**: Using `__threadfence()` + `atomicInc(&count)` allows the last block to perform final reduction without launching a second kernel, eliminating kernel launch overhead.

4. **Optimized memory access**: Sequential addressing (Reduction #3 pattern) ensures no bank conflicts in shared memory, while the cascaded approach minimizes shared memory writes by accumulating in registers first.

**Block/Grid Configuration Rationale**:

We used persistent blocks where GRID_SIZE = 34 SMs × 24 blocks/SM = 816 blocks, and BLOCK_SIZE = 1536 threads/SM ÷ 24 blocks/SM = 64 threads per block. This configuration ensures maximum occupancy on the RTX 4060 Ti while keeping enough threads per block for efficient warp-level reduction.
