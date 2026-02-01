# Homework 10 - Discussion
For this Ben Olschar, 108021211678 and Frederik Hüttemann, 108021215247 cooperated with eachother. 
We did implement the code on our own, but agreed to using one version. The discussion and results are made in cooperation.

# Task Configuration
We use a fixed Full HD image size (1920 x 1080) for all tests. This represents a realistic image processing workload. The image data is stored as single-precision floats (4 bytes per pixel), resulting in approximately 8 MB per image.

# Task 0: Execution Time Measurements
In Task 0, we measured the individual execution times for each component of the image processing pipeline. The three main phases are:  
- **Host-to-Device (H2D) transfer**: 1.44 ms  
- **Kernel execution**: 1.47 ms   
- **Device-to-Host (D2H) transfer**: 1.35 ms

These measurements are well-balanced (~1.4-1.5 ms each), which is important for efficient stream-based parallelization. Since the three phases have similar execution times, there is good potential for overlapping operations across multiple CUDA streams.

# Task 1: Sequential Processing (No Streaming)
In Task 1, we processed 100 Full HD images sequentially without utilizing CUDA streams. Each image goes through the complete pipeline: H2D transfer, kernel execution, and D2H transfer. The average execution time was **416.03 ms** across 10 runs. This establishes our baseline for comparison with the streaming approach.

The execution time per image is approximately 4.16 ms, which matches the sum of the individual phases (~1.44 + 1.47 + 1.35 = 4.26 ms from Task 0). The minimal overhead demonstrates efficient synchronization and memory management in the sequential implementation.

# Task 2: Streaming with Overlap
In Task 2, we process 100 Full HD images using CUDA streams to enable overlapping of H2D transfers, kernel execution, and D2H transfers. We used 4 streams to allow for multiple images to be at different stages of the pipeline simultaneously. The average execution time was **380.02 ms** (excluding the warmup run), achieving a speedup of **1.095x (9.5%)** compared to the sequential approach.

The limited speedup can be explained by analyzing the phase durations. Since all three phases (H2D, kernel, D2H) have nearly equal execution times (~1.4-1.5 ms), the pipeline can only partially overlap operations. In an ideal scenario with perfectly balanced phases, we could achieve up to a 2x speedup with 2 streams. However, the GPU's internal scheduling and memory bandwidth limitations prevent complete overlap. Additionally, with 4 streams and 100 images (25 images per stream), we have sufficient pipeline stages, but the balanced phase durations limit the achievable parallelism.

The warmup run (570.15 ms) shows significantly higher latency, likely due to kernel compilation and GPU initialization overhead, which is why it was excluded from the average.

